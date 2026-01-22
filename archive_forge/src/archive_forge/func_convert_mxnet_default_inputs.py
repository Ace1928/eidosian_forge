from typing import Any, Callable, Optional, Tuple, Type
from ..config import registry
from ..model import Model
from ..shims import MXNetShim
from ..types import ArgsKwargs
from ..util import convert_recursive, is_mxnet_array, is_xp_array, mxnet2xp, xp2mxnet
def convert_mxnet_default_inputs(model: Model, X: Any, is_train: bool) -> Tuple[ArgsKwargs, Callable[[ArgsKwargs], Any]]:
    xp2mxnet_ = lambda x: xp2mxnet(x, requires_grad=is_train)
    converted = convert_recursive(is_xp_array, xp2mxnet_, X)
    if isinstance(converted, ArgsKwargs):

        def reverse_conversion(dXmxnet):
            return convert_recursive(is_mxnet_array, mxnet2xp, dXmxnet)
        return (converted, reverse_conversion)
    elif isinstance(converted, dict):

        def reverse_conversion(dXmxnet):
            dX = convert_recursive(is_mxnet_array, mxnet2xp, dXmxnet)
            return dX.kwargs
        return (ArgsKwargs(args=tuple(), kwargs=converted), reverse_conversion)
    elif isinstance(converted, (tuple, list)):

        def reverse_conversion(dXmxnet):
            dX = convert_recursive(is_mxnet_array, mxnet2xp, dXmxnet)
            return dX.args
        return (ArgsKwargs(args=tuple(converted), kwargs={}), reverse_conversion)
    else:

        def reverse_conversion(dXmxnet):
            dX = convert_recursive(is_mxnet_array, mxnet2xp, dXmxnet)
            return dX.args[0]
        return (ArgsKwargs(args=(converted,), kwargs={}), reverse_conversion)