import collections
import dataclasses
import functools
import inspect
import itertools
import operator
import sys
import types
from typing import Dict, List
import torch._C
import torch._numpy as tnp
from .. import config, polyfill, variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..exc import unimplemented
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, GetItemSource, ODictGetItemSource, TypeSource
from ..utils import (
from .base import MutableLocal, VariableTracker
from .dicts import DefaultDictVariable
from .functions import (
from .user_defined import UserDefinedObjectVariable
def const_getattr(self, tx, name):
    if not isinstance(self.obj, variables.NNModuleVariable):
        raise NotImplementedError()
    step1 = tx.output.get_submodule(self.obj.module_key)
    if self.name not in step1.__dict__:
        raise NotImplementedError()
    step2 = inspect.getattr_static(step1, self.name)
    if name not in step2.__dict__:
        raise NotImplementedError()
    return inspect.getattr_static(step2, name)