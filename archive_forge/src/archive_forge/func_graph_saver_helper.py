import copy
import logging
import os
import pickle
import random
from contextlib import contextmanager
from functools import partial
from typing import Callable, Union
import sympy
import torch
from torch import SymInt
import torch.fx as fx
import torch.nn as nn
from torch._decomp import get_decompositions
from torch.fx.experimental.symbolic_shapes import bind_symbols
from .aot_autograd import aot_function, aot_module, make_boxed_compiler
from .compile_utils import strip_overloads
from .partitioners import (
import torch.utils._pytree as pytree
import torch
import torch.fx as fx
from functorch.compile import minifier, check_nvfuser_subprocess, check_nvfuser_correctness_subprocess
from foo import FxModule
def graph_saver_helper(gm_to_save, args, type_name):
    global graph_index
    if len(gm_to_save.graph.nodes) == 0:
        log.log(logging.WARNING, 'No nodes in graph {%s}_{%s}_{%s}.', current_name, type_name, graph_index)
        return
    gm = copy.deepcopy(gm_to_save)
    gm.graph.set_codegen(torch.fx.graph.CodeGen())
    gm.recompile()
    input_meta = get_input_meta(args)
    isExist = os.path.exists(f'{folder_name}/{current_name}')
    if not isExist:
        os.makedirs(f'{folder_name}/{current_name}')
    gm.to_folder(f'{folder_name}/{current_name}/{current_name}_{type_name}_{graph_index}')
    pickle.dump(input_meta, open(f'{folder_name}/{current_name}/{current_name}_{type_name}_{graph_index}/{current_name}_{type_name}_{graph_index}.input', 'wb'))
    if dump_example_input:
        torch.save(args, f'{folder_name}/{current_name}/{current_name}_{type_name}_{graph_index}/{current_name}_{type_name}_{graph_index}.pt')