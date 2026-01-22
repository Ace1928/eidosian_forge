import ast
import collections
import io
import sys
import token
import tokenize
from abc import ABCMeta
from ast import Module, expr, AST
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union, cast, Any, TYPE_CHECKING
from six import iteritems
def match_token(token, tok_type, tok_str=None):
    """Returns true if token is of the given type and, if a string is given, has that string."""
    return token.type == tok_type and (tok_str is None or token.string == tok_str)