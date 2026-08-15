"""The plugin hook takes the call the dispatch makes, and the call the plugins
installed before the parameter moved still make.

`CustomSpecAlgo` is the out-of-tree extension point: a registered algorithm's
method is called through the same dispatch as the built-in ones, and nothing in
the tree implements it, so a drift between the two sides only ever surfaces in
somebody's plugin. It has drifted twice. The built-in disaggregation
draft-input builder dropped its config parameter while the hook kept it, so
every plugin call would have hit a TypeError; dropping it from the hook too
breaks the other direction, because an installed plugin passes the argument and
an installed plugin's override is defined to receive it.

What is pinned here:

  * every method the dispatch may call on either type takes the same arguments
    on both -- the set is intersected out of the two types, and the parameters
    a removal marker in `spec_registry` names are dropped from the hook side;
  * the hook binds the call the dispatch actually writes, and the pre-bag call,
    each to the same arguments, and only the pre-bag call warns;
  * an override written against the pre-bag argument list still takes the
    dispatch's call, with the arguments in its own slots;
  * the compatibility window named in the marker has not closed -- when it has,
    this test is the reminder to delete the shim.
"""

import ast
import inspect
import re
import unittest
import warnings
from datetime import date
from pathlib import Path
from typing import NamedTuple

from sglang.srt.disaggregation import decode_schedule_batch_mixin
from sglang.srt.runtime_context import get_context, reset_context
from sglang.srt.speculative import spec_registry
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_registry import CustomSpecAlgo
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# The pre-bag call, fixed by the plugins already installed against it.
_LEGACY_POSITIONAL = ("batch", "server_args", "last_tokens_tensor", "future_map")

_MARKER = re.compile(
    r"TODO\((?P<owner>[\w.@-]+), (?P<deadline>\d{4}-\d{2}-\d{2})\): remove the "
    r"deprecated ``(?P<parameter>\w+)`` parameter"
)


class _Marker(NamedTuple):
    method: str
    parameter: str
    deadline: date
    owner: str


def _removal_markers():
    """The compatibility shims `spec_registry` carries, read from the markers
    that retire them: owner, date, parameter, and the method the marker sits
    on. Read out of the source so the shim is named once, in the code."""
    source = Path(spec_registry.__file__).read_text(encoding="utf-8")
    definitions = sorted(
        (node.lineno, node.name)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    markers = {}
    for lineno, line in enumerate(source.splitlines(), start=1):
        match = _MARKER.search(line)
        if match is None:
            continue
        method = next(name for def_line, name in definitions if def_line > lineno)
        markers[method] = _Marker(
            method=method,
            parameter=match["parameter"],
            deadline=date.fromisoformat(match["deadline"]),
            owner=match["owner"],
        )
    return markers


def _dispatched_methods():
    """Methods carried by both types. The dispatch calls them on whichever it
    holds without knowing which, so their argument lists must agree."""
    enum_methods = {
        name
        for name, value in vars(SpeculativeAlgorithm).items()
        if inspect.isfunction(value)
    }
    hook_methods = {
        name
        for name, value in vars(CustomSpecAlgo).items()
        if inspect.isfunction(value)
    }
    return sorted(enum_methods & hook_methods)


def _dispatch_calls():
    """Every call the decode dispatch makes on the algorithm object, read from
    its source: `(method, positional count, keyword names)`."""
    source = Path(decode_schedule_batch_mixin.__file__).read_text(encoding="utf-8")
    calls = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "spec_algorithm"
        ):
            continue
        calls.append((func.attr, len(node.args), tuple(kw.arg for kw in node.keywords)))
    return calls


def _parameters(function, dropped=frozenset()):
    """Parameter names, without the shim's catch-alls and `dropped`."""
    return [
        name
        for name, parameter in inspect.signature(function).parameters.items()
        if parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        and name not in dropped
    ]


def _algo():
    return CustomSpecAlgo(name="TEST_HOOK", factory=lambda server_args: object)


class TestDispatchedSignatures(CustomTestCase):
    def test_the_hook_and_the_built_in_agree(self):
        markers = _removal_markers()
        methods = _dispatched_methods()
        self.assertNotEqual(methods, [], "the dispatched set derived to nothing")
        mismatches = []
        for name in methods:
            marker = markers.get(name)
            dropped = frozenset([marker.parameter]) if marker else frozenset()
            hook = _parameters(getattr(CustomSpecAlgo, name), dropped)
            builtin = _parameters(getattr(SpeculativeAlgorithm, name))
            if hook != builtin:
                mismatches.append(
                    f"{name}: CustomSpecAlgo{tuple(hook)} vs "
                    f"SpeculativeAlgorithm{tuple(builtin)}"
                )
        self.assertEqual(
            mismatches,
            [],
            "a plugin implementing the hook would be called with the "
            "dispatch's arguments:\n  " + "\n  ".join(mismatches),
        )

    def test_the_dispatch_call_binds_on_both_types(self):
        calls = _dispatch_calls()
        self.assertNotEqual(calls, [], "no dispatch call found to bind against")
        for method, positional, keywords in calls:
            self.assertIn(method, _dispatched_methods())
            for owner in (CustomSpecAlgo, SpeculativeAlgorithm):
                arguments = [None] * (1 + positional)
                inspect.signature(getattr(owner, method)).bind(
                    *arguments, **{name: None for name in keywords}
                )


class TestDeprecatedServerArgs(CustomTestCase):
    """The `server_args` compatibility shim on `build_disagg_draft_input`."""

    def setUp(self):
        super().setUp()
        self._saved_server_args = get_context()._server_args

    def tearDown(self):
        if self._saved_server_args is None:
            reset_context()
        else:
            get_context().set_server_args(self._saved_server_args)
        super().tearDown()

    @property
    def _marker(self):
        markers = _removal_markers()
        self.assertIn(
            "build_disagg_draft_input",
            markers,
            "the hook carries a server_args shim with no removal marker",
        )
        return markers["build_disagg_draft_input"]

    def _dispatch_call_shape(self):
        calls = [
            call for call in _dispatch_calls() if call[0] == "build_disagg_draft_input"
        ]
        self.assertEqual(len(calls), 1, f"expected one dispatch call, got {calls}")
        return calls[0]

    def test_the_marker_names_the_parameter_the_hook_carries(self):
        marker = self._marker
        parameter = inspect.signature(
            getattr(CustomSpecAlgo, marker.method)
        ).parameters.get(marker.parameter)
        self.assertIsNotNone(
            parameter, f"{marker.method}() has no {marker.parameter} parameter"
        )
        self.assertIs(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIsNone(
            parameter.default,
            "the deprecated parameter must default to None so its absence is "
            "what a current call looks like",
        )

    def test_the_dispatch_call_does_not_warn(self):
        _, positional, keywords = self._dispatch_call_shape()
        arguments = [f"arg{i}" for i in range(positional)]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = _algo().build_disagg_draft_input(
                *arguments, **{name: None for name in keywords}
            )
        self.assertIsNone(result)
        self.assertEqual([str(entry.message) for entry in caught], [])

    def test_the_legacy_call_warns(self):
        marker = self._marker
        with self.assertWarns(DeprecationWarning) as caught:
            result = _algo().build_disagg_draft_input(*_LEGACY_POSITIONAL)
        self.assertIsNone(result)
        self.assertIn(marker.parameter, str(caught.warning))
        self.assertIn("future release", str(caught.warning))
        # The warning blames the caller, not a frame inside sglang.
        self.assertEqual(Path(caught.filename).resolve(), Path(__file__).resolve())

    def test_a_legacy_override_takes_both_calls(self):
        seen = {}

        with self.assertWarns(DeprecationWarning) as caught:

            class LegacyPlugin(CustomSpecAlgo):
                def build_disagg_draft_input(
                    self, batch, server_args, last_tokens_tensor, future_map
                ):
                    seen.update(
                        batch=batch,
                        server_args=server_args,
                        last_tokens_tensor=last_tokens_tensor,
                        future_map=future_map,
                    )
                    return "spec_info"

        # The warning blames the plugin's class statement.
        self.assertEqual(Path(caught.filename).resolve(), Path(__file__).resolve())
        algo = LegacyPlugin(name="TEST_LEGACY", factory=lambda server_args: object)

        _, positional, keywords = self._dispatch_call_shape()
        arguments = [f"arg{i}" for i in range(positional)]
        with warnings.catch_warnings(record=True) as at_call:
            warnings.simplefilter("always")
            result = algo.build_disagg_draft_input(
                *arguments, **{name: None for name in keywords}
            )
        self.assertEqual(result, "spec_info")
        self.assertEqual([str(entry.message) for entry in at_call], [])
        # Every argument the dispatch passes lands in its pre-bag slot; the slot
        # the dispatch no longer fills stays empty rather than being filled from
        # the process record, so the override runs in a process that has not
        # published and reads the bag for what it used to take from that object.
        dispatched = dict(
            zip(
                [name for name in _LEGACY_POSITIONAL if name != "server_args"],
                arguments,
            )
        )
        self.assertEqual(seen, {**dispatched, "server_args": None})

        seen.clear()
        with self.assertWarns(DeprecationWarning) as caught:
            self.assertEqual(
                algo.build_disagg_draft_input(*_LEGACY_POSITIONAL), "spec_info"
            )
        self.assertEqual(Path(caught.filename).resolve(), Path(__file__).resolve())
        # The pre-bag call keeps reaching the override argument for argument,
        # its own server_args included.
        self.assertEqual(seen, dict(zip(_LEGACY_POSITIONAL, _LEGACY_POSITIONAL)))


if __name__ == "__main__":
    unittest.main()
