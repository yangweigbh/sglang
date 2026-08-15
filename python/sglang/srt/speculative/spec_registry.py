"""Internal storage backing ``SpeculativeAlgorithm.register``. Plugins
should use that classmethod API; do not import from this module directly.
"""

from __future__ import annotations

import inspect
import logging
import warnings
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple, Type

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.spec_info import SpecInput

WorkerFactory = Callable[["ServerArgs"], Type]
ServerArgsValidator = Callable[["ServerArgs"], None]

logger = logging.getLogger(__name__)


class CustomSpecAlgo:
    """A plugin-registered speculative algorithm. Duck-types
    ``SpeculativeAlgorithm`` enum values (same ``is_*()`` / ``create_worker``
    interface).

    Plugins may subclass this to override any ``is_*()`` / ``supports_*()`` /
    ``create_worker`` method (e.g. to integrate with builtin-specific
    branches like ``if spec_algorithm.is_eagle():`` in scheduler /
    model_runner). Pass the subclass via ``spec_class=...`` at registration.

    Defaults: all ``is_*()`` return ``False`` except ``is_speculative``.

    ``supports_overlap=False`` is deprecated: the spec V1 worker path has been
    removed, so such algorithms run on the V2 scheduler schema with overlap
    disabled (synchronous). Migrate plugin workers to the V2 schema and
    overlap scheduling.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        override = cls.__dict__.get("build_disagg_draft_input")
        if not inspect.isfunction(override) or not _is_legacy_disagg_override(
            inspect.signature(override)
        ):
            return
        warnings.warn(
            f"{cls.__module__}.{cls.__qualname__}.build_disagg_draft_input() "
            "takes the deprecated server_args argument; it is called without "
            "one and the argument will be removed in a future release. Define "
            "it as build_disagg_draft_input(self, batch, last_tokens_tensor, "
            "future_map) and read the speculative config from "
            "sglang.srt.runtime_context.get_spec().",
            DeprecationWarning,
            # This method, then the class statement that triggered it.
            stacklevel=2,
        )
        cls.build_disagg_draft_input = _adapt_legacy_disagg_override(override)

    def __init__(
        self,
        name: str,
        factory: WorkerFactory,
        *,
        supports_overlap: bool = False,
        validate_server_args: Optional[ServerArgsValidator] = None,
    ):
        self.name = name
        self.factory = factory
        self.supports_overlap = supports_overlap
        self.validate_server_args = validate_server_args

    def __repr__(self) -> str:
        return f"CustomSpecAlgo({self.name!r})"

    def is_some(self) -> bool:
        return True

    def is_none(self) -> bool:
        return False

    def is_speculative(self) -> bool:
        return True

    def is_eagle(self) -> bool:
        return False

    def is_eagle3(self) -> bool:
        return False

    def is_frozen_kv_mtp(self) -> bool:
        return False

    def is_dflash(self) -> bool:
        return False

    def is_dspark(self) -> bool:
        return False

    def is_dflash_family(self) -> bool:
        return False

    def is_standalone(self) -> bool:
        return False

    def is_ngram(self) -> bool:
        return False

    def supports_target_verify_for_draft(self) -> bool:
        return False

    def is_last_shared_read_phase(self, forward_mode) -> bool:
        # The step's last shared-buffer-reading phase owns the shared-read-done publish.
        return forward_mode.is_draft_extend_v2()

    def supports_ragged_verify(self) -> bool:
        return False

    def supports_grammar_overlap(self) -> bool:
        # Whether the worker advances the grammar FSM inside verify() (via the
        # scheduler's grammar barrier), letting spec + grammar decode overlap.
        return False

    def has_draft_kv(self) -> bool:
        # Conservative default: the larger KV reserve.
        return True

    def handle_server_args(self, server_args: ServerArgs) -> None:
        pass

    def create_worker(self, server_args: ServerArgs) -> Type:
        if not server_args.disable_overlap_schedule and not self.supports_overlap:
            raise ValueError(
                f"Speculative algorithm {self.name} does not support overlap scheduling."
            )
        if not self.supports_overlap:
            # Reached only when overlap is disabled, so the algorithm really
            # does run synchronously on the V2 schema below.
            logger.warning(
                "Speculative algorithm %s is registered with "
                "supports_overlap=False, which is deprecated: the spec V1 "
                "worker path has been removed, and the algorithm now runs on "
                "the V2 scheduler schema with overlap disabled (synchronous). "
                "Migrate the plugin worker to support overlap scheduling.",
                self.name,
            )
        return self.factory(server_args)

    def get_num_tokens_per_req_for_target_verify(
        self, num_draft_tokens: int, is_draft_worker: bool
    ) -> int:
        # FIXME: Remove this after the forward mode refactor. Target verify is
        # essentially a fixed sequence length prefill/extend with full cuda
        # graph support. We can use it for target verify, or we can use it for
        # other cases which is not target verify but fixed length prefill.
        # Here, we expose this interface to allow the other use cases.
        return num_draft_tokens

    def get_num_tokens_per_bs_for_target_verify(
        self, num_draft_tokens: int, is_draft_worker: bool
    ) -> int:
        # Deprecated alias; remove together with the FIXME above.
        warnings.warn(
            "get_num_tokens_per_bs_for_target_verify is deprecated; use "
            "get_num_tokens_per_req_for_target_verify instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_num_tokens_per_req_for_target_verify(
            num_draft_tokens, is_draft_worker
        )

    # TODO(ch-wan, 2026-09-17): remove the deprecated ``server_args`` parameter
    # below, together with the ``*args`` shim and the legacy-override adapter;
    # the hook is then ``(self, batch, last_tokens_tensor, future_map)``.
    def build_disagg_draft_input(
        self,
        batch: ScheduleBatch,
        *args,
        server_args: Optional[ServerArgs] = None,
        last_tokens_tensor: Optional[torch.Tensor] = None,
        future_map: Optional[FutureMap] = None,
    ) -> Optional[SpecInput]:
        """Build the disaggregation draft input for ``batch``, or ``None``.

        The call is ``(batch, last_tokens_tensor, future_map)``. ``server_args``
        is bound only by the pre-bag call shape and is not read here: the
        speculative config comes from ``runtime_context.get_spec()``, which
        follows a runtime override where the startup record does not. An
        override still written against the pre-bag shape keeps working and is
        handed ``server_args=None`` under the current call -- read the bag for
        any value it used to take from that object.
        """
        _resolve_disagg_draft_input_args(
            args, server_args, last_tokens_tensor, future_map
        )
        return None


# The pre-bag call passed ``server_args`` in the position the current call uses
# for ``last_tokens_tensor``, so the two shapes are separated by how many
# positional arguments follow ``batch`` -- never by looking at the values.
_LEGACY_DISAGG_POSITIONAL = ("server_args", "last_tokens_tensor", "future_map")
_DISAGG_POSITIONAL = ("last_tokens_tensor", "future_map")


def _resolve_disagg_draft_input_args(
    args: tuple,
    server_args: Optional[ServerArgs],
    last_tokens_tensor: Optional[torch.Tensor],
    future_map: Optional[FutureMap],
) -> Tuple[Optional[ServerArgs], Optional[torch.Tensor], Optional[FutureMap]]:
    """Bind ``build_disagg_draft_input`` arguments from either call shape.

    ``args`` are the positional arguments after ``batch``: three is the pre-bag
    shape, two the current one. A call that carries ``server_args`` gets the
    deprecation warning; the argument itself is passed through for the adapter
    below and read nowhere else.
    """
    bound = {
        "server_args": server_args,
        "last_tokens_tensor": last_tokens_tensor,
        "future_map": future_map,
    }
    if len(args) == len(_LEGACY_DISAGG_POSITIONAL):
        names = _LEGACY_DISAGG_POSITIONAL
    elif len(args) == len(_DISAGG_POSITIONAL):
        names = _DISAGG_POSITIONAL
    elif len(args) == 1:
        # One positional plus keywords: the lone positional fills the first slot
        # the keywords left open, which is ``server_args`` exactly when the
        # caller named ``last_tokens_tensor`` itself.
        names = (
            ("server_args",) if last_tokens_tensor is not None else _DISAGG_POSITIONAL
        )
    elif not args:
        names = ()
    else:
        raise TypeError(
            "build_disagg_draft_input() takes (batch, last_tokens_tensor, "
            f"future_map); got {1 + len(args)} positional arguments"
        )
    for name, value in zip(names, args):
        if bound[name] is not None:
            raise TypeError(
                f"build_disagg_draft_input() got multiple values for argument "
                f"'{name}'"
            )
        bound[name] = value
    if bound["server_args"] is not None:
        warnings.warn(
            "Passing server_args to CustomSpecAlgo.build_disagg_draft_input() "
            "is deprecated and will be removed in a future release. Call it as "
            "build_disagg_draft_input(batch, last_tokens_tensor, future_map) "
            "and read the speculative config from "
            "sglang.srt.runtime_context.get_spec().",
            DeprecationWarning,
            # This helper, the hook that calls it, then the hook's caller.
            stacklevel=3,
        )
    return bound["server_args"], bound["last_tokens_tensor"], bound["future_map"]


def _binds_positionally(signature: inspect.Signature, count: int) -> bool:
    """Whether ``signature`` accepts ``count`` positional arguments."""
    try:
        signature.bind(*([None] * count))
    except TypeError:
        return False
    return True


def _is_legacy_disagg_override(signature: inspect.Signature) -> bool:
    """Whether an override is written against the pre-bag argument list.

    Pre-bag means it takes ``self`` plus four positional arguments. An override
    that also binds the current three-argument call is pre-bag only when the
    slot the current call fills with ``last_tokens_tensor`` is named
    ``server_args``, which is how a trailing default (``future_map=None``)
    would otherwise mis-bind in silence.
    """
    if not _binds_positionally(signature, 5):
        return False
    if not _binds_positionally(signature, 4):
        return True
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) > 2 and positional[2].name == "server_args"


def _adapt_legacy_disagg_override(override: Callable) -> Callable:
    """Wrap a pre-bag override so the dispatch's call reaches it."""

    def build_disagg_draft_input(
        self,
        batch,
        *args,
        server_args=None,
        last_tokens_tensor=None,
        future_map=None,
    ):
        server_args, last_tokens_tensor, future_map = _resolve_disagg_draft_input_args(
            args, server_args, last_tokens_tensor, future_map
        )
        # ``server_args`` stays whatever the caller bound: the current call
        # shape binds nothing, so a pre-bag override is handed ``None``. The
        # shim does not reach for the process-wide record -- an override that
        # needs a resolved value reads the bag for it, which is the value a
        # runtime override moves and the record does not.
        return override(self, batch, server_args, last_tokens_tensor, future_map)

    build_disagg_draft_input.__doc__ = override.__doc__
    build_disagg_draft_input.__qualname__ = override.__qualname__
    build_disagg_draft_input.__module__ = override.__module__
    build_disagg_draft_input._legacy_disagg_override = override
    return build_disagg_draft_input


_REGISTRY: Dict[str, CustomSpecAlgo] = {}

# CLI spellings that are not ``SpeculativeAlgorithm`` members but still resolve
# to a builtin (e.g. NEXTN -> EAGLE). Reserved alongside the enum members so
# plugins cannot shadow them.
_RESERVED_ALIASES = frozenset({"NEXTN"})


def _reserved_names() -> frozenset:
    """Names plugins cannot register under: every ``SpeculativeAlgorithm``
    member plus ``_RESERVED_ALIASES``.

    Derived from the enum (lazily, to avoid a circular import — ``spec_info``
    imports this module) so any new builtin is reserved automatically without
    editing a second list.
    """
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    return frozenset(algo.name for algo in SpeculativeAlgorithm) | _RESERVED_ALIASES


def _assert_custom_spec_algo_conforms(spec_class: Type[CustomSpecAlgo]) -> None:
    """Fail fast if ``spec_class`` drifts from the ``SpeculativeAlgorithm``
    duck-typing contract.

    ``from_string`` returns either type and callers dispatch on the shared
    ``is_*()`` / ``supports_*()`` interface without isinstance checks, so every
    such method on the enum must also exist on the registered spec class —
    otherwise a plugin-registered algo hits ``AttributeError`` at a call site
    (this is how ``is_some`` / ``is_frozen_kv_mtp`` silently went missing). New
    predicates are covered automatically; no second list to maintain.

    Called from ``register_algorithm`` rather than at import time because
    ``spec_info`` imports this module, so ``SpeculativeAlgorithm`` does not yet
    exist while this module is loading; at registration time it is fully
    defined.
    """
    # NOTE: use ``vars()`` not ``dir()`` for the enum — ``EnumMeta.__dir__``
    # hides instance methods, so ``dir(SpeculativeAlgorithm)`` would yield an
    # empty interface and turn this guard into a silent no-op.
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    interface = {
        name
        for name in vars(SpeculativeAlgorithm)
        if name.startswith(("is_", "supports_"))
    }
    missing = sorted(interface - set(dir(spec_class)))
    if missing:
        raise TypeError(
            f"{spec_class.__name__} is missing duck-typed methods from "
            f"SpeculativeAlgorithm: {missing}. Add them to {spec_class.__name__} "
            "so plugin-registered algorithms stay dispatchable."
        )


def register_algorithm(
    name: str,
    *,
    supports_overlap: bool = False,
    validate_server_args: Optional[ServerArgsValidator] = None,
    spec_class: Type[CustomSpecAlgo] = CustomSpecAlgo,
) -> Callable[[WorkerFactory], WorkerFactory]:
    """Return a decorator that registers a plugin algorithm under ``name``.

    Pass a ``spec_class`` subclass of ``CustomSpecAlgo`` to override any
    ``is_*()`` / ``supports_*()`` / ``create_worker`` method.
    """
    upper = name.upper()
    if upper in _reserved_names():
        raise ValueError(
            f"'{upper}' is a reserved speculative algorithm name; cannot be re-registered."
        )
    if upper in _REGISTRY:
        raise ValueError(f"Speculative algorithm '{upper}' already registered.")
    _assert_custom_spec_algo_conforms(spec_class)

    def decorator(factory: WorkerFactory) -> WorkerFactory:
        _REGISTRY[upper] = spec_class(
            name=upper,
            factory=factory,
            supports_overlap=supports_overlap,
            validate_server_args=validate_server_args,
        )
        return factory

    return decorator


def get_spec(name: Optional[str]) -> Optional[CustomSpecAlgo]:
    """Return the registered spec for ``name``, or ``None`` for builtin /
    unknown names."""
    if name is None:
        return None
    return _REGISTRY.get(name.upper())
