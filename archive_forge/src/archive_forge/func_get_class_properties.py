import ast
import dataclasses
import inspect
import re
import string
import sys
from collections import namedtuple
from textwrap import dedent
from typing import List, Tuple  # noqa: F401
import torch
import torch.jit.annotations
from torch import _jit_internal
from torch._C._jit_tree_views import (
from torch._jit_internal import (  # noqa: F401
from torch._sources import (
from torch.jit._dataclass_impls import DATACLASS_MAGIC_METHODS
from torch.jit._monkeytype_config import get_qualified_name, monkeytype_trace
def get_class_properties(cls, self_name):
    """
    Get a list of Property objects representing the properties of a class.

    Args:
        cls:  The class to get properties of.
        self_name: The name of the class that the properties should belong to.
    Returns:
        A list of Property objects corresponding to the properties of cls. Property
        here refers to the subclass of TreeView.
    """
    props = inspect.getmembers(cls, predicate=lambda m: isinstance(m, property))
    unused_properties = getattr(cls, '__jit_unused_properties__', [])
    properties = []
    for prop in props:
        if prop[0] not in unused_properties and (not should_drop(prop[1].fget)):
            getter = get_jit_def(prop[1].fget, f'__{prop[0]}_getter', self_name=self_name)
            setter = get_jit_def(prop[1].fset, f'__{prop[0]}_setter', self_name=self_name) if prop[1].fset else None
            properties.append(Property(getter.range(), Ident(getter.range(), prop[0]), getter, setter))
    return properties