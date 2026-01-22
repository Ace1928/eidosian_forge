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
@functools.lru_cache(maxsize=1)
def get_np_to_tnp_map():
    from ..utils import NP_TO_TNP_MODULE
    np_fn_to_tnp_fn = {}
    for np_mod, tnp_mod in NP_TO_TNP_MODULE.items():
        for fn_name, tnp_fn in tnp_mod.__dict__.items():
            if callable(tnp_fn):
                if (np_fn := getattr(np_mod, fn_name, None)):
                    np_fn_to_tnp_fn[np_fn] = tnp_fn
    return np_fn_to_tnp_fn