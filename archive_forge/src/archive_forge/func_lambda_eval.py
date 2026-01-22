from __future__ import annotations
import builtins
from types import CodeType
from typing import Any, Callable
from . import Image, _imagingmath
from ._deprecate import deprecate
def lambda_eval(expression: Callable[[dict[str, Any]], Any], options: dict[str, Any]={}, **kw: Any) -> Any:
    """
    Returns the result of an image function.

    :py:mod:`~PIL.ImageMath` only supports single-layer images. To process multi-band
    images, use the :py:meth:`~PIL.Image.Image.split` method or
    :py:func:`~PIL.Image.merge` function.

    :param expression: A function that receives a dictionary.
    :param options: Values to add to the function's dictionary. You
                    can either use a dictionary, or one or more keyword
                    arguments.
    :return: The expression result. This is usually an image object, but can
             also be an integer, a floating point value, or a pixel tuple,
             depending on the expression.
    """
    args: dict[str, Any] = ops.copy()
    args.update(options)
    args.update(kw)
    for k, v in args.items():
        if hasattr(v, 'im'):
            args[k] = _Operand(v)
    out = expression(args)
    try:
        return out.im
    except AttributeError:
        return out