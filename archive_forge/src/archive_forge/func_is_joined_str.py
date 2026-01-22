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
def is_joined_str(node):
    """Returns whether node is a JoinedStr node, used to represent f-strings."""
    return node.__class__.__name__ == 'JoinedStr'