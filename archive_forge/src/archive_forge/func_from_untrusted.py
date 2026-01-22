import inspect
import operator
import typing as t
from collections import deque
from markupsafe import Markup
from .utils import _PassArg
@classmethod
def from_untrusted(cls, value: t.Any, lineno: t.Optional[int]=None, environment: 't.Optional[Environment]'=None) -> 'Const':
    """Return a const object if the value is representable as
        constant value in the generated code, otherwise it will raise
        an `Impossible` exception.
        """
    from .compiler import has_safe_repr
    if not has_safe_repr(value):
        raise Impossible()
    return cls(value, lineno=lineno, environment=environment)