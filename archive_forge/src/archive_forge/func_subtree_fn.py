from collections import abc as collections_abc
import logging
import sys
from typing import Mapping, Sequence, Text, TypeVar, Union
from .sequence import _is_attrs
from .sequence import _is_namedtuple
from .sequence import _sequence_like
from .sequence import _sorted
def subtree_fn(item):
    subtree_path, subtree = item
    return traverse_impl(path + (subtree_path,), subtree)