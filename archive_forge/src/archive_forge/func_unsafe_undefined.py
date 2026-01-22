import operator
import types
import typing as t
from _string import formatter_field_name_split  # type: ignore
from collections import abc
from collections import deque
from string import Formatter
from markupsafe import EscapeFormatter
from markupsafe import Markup
from .environment import Environment
from .exceptions import SecurityError
from .runtime import Context
from .runtime import Undefined
def unsafe_undefined(self, obj: t.Any, attribute: str) -> Undefined:
    """Return an undefined object for unsafe attributes."""
    return self.undefined(f'access to attribute {attribute!r} of {type(obj).__name__!r} object is unsafe.', name=attribute, obj=obj, exc=SecurityError)