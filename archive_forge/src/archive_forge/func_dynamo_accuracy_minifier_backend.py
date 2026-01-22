import argparse
import copy
import functools
import logging
import os
import shutil
import sys
import textwrap
from importlib import import_module
from typing import Union
import torch
import torch.fx as fx
from torch._dynamo.debug_utils import (
from torch.fx.experimental.symbolic_shapes import fx_placeholder_targets
from torch.hub import tqdm
from .. import config
from ..backends.registry import lookup_backend, register_debug_backend
from ..debug_utils import clone_inputs_retaining_gradness
from math import inf
import torch
from torch import tensor, device
import torch.fx as fx
import torch._dynamo
from torch._dynamo.testing import rand_strided
from torch._dynamo.debug_utils import run_fwd_maybe_bwd
@register_debug_backend
def dynamo_accuracy_minifier_backend(gm, example_inputs, compiler_name):
    from functorch.compile import minifier
    compiler_fn = lookup_backend(compiler_name)
    gm.eval()
    if backend_accuracy_fails(gm, example_inputs, compiler_fn, only_fwd=config.repro_forward_only):
        log.warning('Accuracy failed for the TorchDynamo produced graph')
        dump_state_fn = functools.partial(dump_backend_state, compiler_name=compiler_name, check_accuracy=True)
        fails_fn = functools.partial(backend_accuracy_fails, compiler_fn=compiler_fn, only_fwd=config.repro_forward_only)
        dump_state_fn(fx.GraphModule(gm, copy.deepcopy(gm.graph)), example_inputs)
        minifier(gm, example_inputs, module_fails=fails_fn, dump_state=dump_state_fn)
    else:
        log.error('Input graph does not fail accuracy testing')
    return gm