import argparse
import copy
import functools
import io
import logging
import os
import shutil
import subprocess
import sys
import textwrap
import uuid
from importlib import import_module
from tempfile import TemporaryFile
from typing import Any, Callable, Dict, Union
import torch
import torch.fx as fx
import torch.nn as nn
from torch._dynamo.debug_utils import (
from torch._dynamo.utils import clone_inputs, counters, same
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import (
from torch.hub import tqdm
from .. import config
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims
def generate_compiler_repro_string(gm, args, *, stable_output=False, save_dir=None):
    model_str = textwrap.dedent(f'\nimport torch\nfrom torch import tensor, device\nimport torch.fx as fx\nfrom torch._dynamo.testing import rand_strided\nfrom math import inf\nimport torch._inductor.inductor_prims\n\n{generate_config_string(stable_output=stable_output)}\n\nisolate_fails_code_str = None\n\n{extra_imports}\n\n        ')
    if not stable_output:
        model_str += f'# torch version: {torch.version.__version__}\n'
        if hasattr(torch.version, 'cuda'):
            model_str += f'# torch cuda version: {torch.version.cuda}\n'
        if hasattr(torch.version, 'git_version'):
            model_str += f'# torch git version: {torch.version.git_version}\n\n\n'
        model_str += _cuda_system_info_comment()
    model_str += NNModuleToString.convert(gm)

    def hint_if_symint(x):
        return tuple((i.node.hint if isinstance(i, torch.SymInt) else i for i in x))
    writer = InputWriter(save_dir)
    for placeholder, arg in zip(fx_placeholder_targets(gm), args):
        if isinstance(arg, (int, torch.SymInt)):
            writer.symint(placeholder, arg)
        elif isinstance(arg, torch.Tensor):
            writer.tensor(placeholder, arg)
        else:
            raise TypeError(f'arg is neither SymInt/int nor torch.Tensor, {arg}')
    model_str += '\n'.join(writer.lines()) + '\n'
    model_str += 'mod = Repro()\n'
    return model_str