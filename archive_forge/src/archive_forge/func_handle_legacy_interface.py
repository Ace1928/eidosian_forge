import functools
import inspect
import warnings
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union
from torch import nn
from .._utils import sequence_to_str
from ._api import WeightsEnum
def handle_legacy_interface(**weights: Tuple[str, Union[Optional[W], Callable[[Dict[str, Any]], Optional[W]]]]):
    """Decorates a model builder with the new interface to make it compatible with the old.

    In particular this handles two things:

    1. Allows positional parameters again, but emits a deprecation warning in case they are used. See
        :func:`torchvision.prototype.utils._internal.kwonly_to_pos_or_kw` for details.
    2. Handles the default value change from ``pretrained=False`` to ``weights=None`` and ``pretrained=True`` to
        ``weights=Weights`` and emits a deprecation warning with instructions for the new interface.

    Args:
        **weights (Tuple[str, Union[Optional[W], Callable[[Dict[str, Any]], Optional[W]]]]): Deprecated parameter
            name and default value for the legacy ``pretrained=True``. The default value can be a callable in which
            case it will be called with a dictionary of the keyword arguments. The only key that is guaranteed to be in
            the dictionary is the deprecated parameter name passed as first element in the tuple. All other parameters
            should be accessed with :meth:`~dict.get`.
    """

    def outer_wrapper(builder: Callable[..., M]) -> Callable[..., M]:

        @kwonly_to_pos_or_kw
        @functools.wraps(builder)
        def inner_wrapper(*args: Any, **kwargs: Any) -> M:
            for weights_param, (pretrained_param, default) in weights.items():
                sentinel = object()
                weights_arg = kwargs.get(weights_param, sentinel)
                if weights_param not in kwargs and pretrained_param not in kwargs or isinstance(weights_arg, WeightsEnum) or (isinstance(weights_arg, str) and weights_arg != 'legacy') or (weights_arg is None):
                    continue
                pretrained_positional = weights_arg is not sentinel
                if pretrained_positional:
                    kwargs[pretrained_param] = pretrained_arg = kwargs.pop(weights_param)
                else:
                    pretrained_arg = kwargs[pretrained_param]
                if pretrained_arg:
                    default_weights_arg = default(kwargs) if callable(default) else default
                    if not isinstance(default_weights_arg, WeightsEnum):
                        raise ValueError(f'No weights available for model {builder.__name__}')
                else:
                    default_weights_arg = None
                if not pretrained_positional:
                    warnings.warn(f"The parameter '{pretrained_param}' is deprecated since 0.13 and may be removed in the future, please use '{weights_param}' instead.")
                msg = f"Arguments other than a weight enum or `None` for '{weights_param}' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `{weights_param}={default_weights_arg}`."
                if pretrained_arg:
                    msg = f'{msg} You can also use `{weights_param}={type(default_weights_arg).__name__}.DEFAULT` to get the most up-to-date weights.'
                warnings.warn(msg)
                del kwargs[pretrained_param]
                kwargs[weights_param] = default_weights_arg
            return builder(*args, **kwargs)
        return inner_wrapper
    return outer_wrapper