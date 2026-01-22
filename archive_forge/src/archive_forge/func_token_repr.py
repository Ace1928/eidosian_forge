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
def token_repr(tok_type, string):
    """Returns a human-friendly representation of a token with the given type and string."""
    return '%s:%s' % (token.tok_name[tok_type], repr(string).lstrip('u'))