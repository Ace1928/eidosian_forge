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
def dump_compiler_graph_state(gm, args, compiler_name, *, accuracy=None):
    subdir = os.path.join(minifier_dir(), 'checkpoints')
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    file_name = os.path.join(subdir, f'{len(gm.graph.nodes)}.py')
    log.warning('Writing checkpoint with %s nodes to %s', len(gm.graph.nodes), file_name)
    with open(file_name, 'w') as fd:
        save_graph_repro(fd, gm, args, compiler_name, save_dir=subdir, accuracy=accuracy)
    curdir = os.getcwd()
    repro_path = os.path.join(curdir, 'repro.py')
    try:
        shutil.copyfile(file_name, repro_path)
        log.warning('Copying repro file for convenience to %s', repro_path)
        if use_buck:
            BuckTargetWriter(file_name).write()
    except OSError:
        log.warning('No write permissions for %s', repro_path)
        pass