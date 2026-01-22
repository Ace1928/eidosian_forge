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
def iter_children_ast(node, include_joined_str=False):
    if not include_joined_str and is_joined_str(node):
        return
    if isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values):
            if key is not None:
                yield key
            yield value
        return
    for child in ast.iter_child_nodes(node):
        if child.__class__ not in SINGLETONS:
            yield child