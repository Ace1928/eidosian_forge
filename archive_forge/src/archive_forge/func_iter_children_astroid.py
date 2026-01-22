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
def iter_children_astroid(node, include_joined_str=False):
    if not include_joined_str and is_joined_str(node):
        return []
    return node.get_children()