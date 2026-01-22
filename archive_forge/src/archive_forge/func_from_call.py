import railroad
from pip._vendor import pyparsing
import typing
from typing import (
from jinja2 import Template
from io import StringIO
import inspect
@classmethod
def from_call(cls, func: Callable[..., T], *args, **kwargs) -> 'EditablePartial[T]':
    """
        If you call this function in the same way that you would call the constructor, it will store the arguments
        as you expect. For example EditablePartial.from_call(Fraction, 1, 3)() == Fraction(1, 3)
        """
    return EditablePartial(func=func, args=list(args), kwargs=kwargs)