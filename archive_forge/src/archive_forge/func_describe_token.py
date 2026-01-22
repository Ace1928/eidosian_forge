import re
import typing as t
from ast import literal_eval
from collections import deque
from sys import intern
from ._identifier import pattern as name_re
from .exceptions import TemplateSyntaxError
from .utils import LRUCache
def describe_token(token: 'Token') -> str:
    """Returns a description of the token."""
    if token.type == TOKEN_NAME:
        return token.value
    return _describe_token_type(token.type)