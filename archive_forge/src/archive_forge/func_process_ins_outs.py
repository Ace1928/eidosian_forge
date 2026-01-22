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
def process_ins_outs(args):
    inputs = []
    outputs = []
    for arg in args:
        var_name = arg.arg
        var_ann = arg.value.value
        var_decl_type, var_ann = var_ann.split(':')
        if var_decl_type == 'inp':
            inputs.append(InputType(var_name, var_ann))
        if var_decl_type == 'out':
            outputs.append(OutputType(var_name, var_ann))
    return (inputs, outputs)