import re
import typing as t
from ast import literal_eval
from collections import deque
from sys import intern
from ._identifier import pattern as name_re
from .exceptions import TemplateSyntaxError
from .utils import LRUCache
def next_if(self, expr: str) -> t.Optional[Token]:
    """Perform the token test and return the token if it matched.
        Otherwise the return value is `None`.
        """
    if self.current.test(expr):
        return next(self)
    return None