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
def compare_tuples(tuple1, tuple2):
    diff_indices = [i for i in range(len(tuple1)) if tuple1[i] != tuple2[i]]
    diff_values = [(tuple1[i], tuple2[i]) for i in diff_indices]
    if not diff_values:
        return None
    else:
        return ' and '.join((f'{a} != {b}' for a, b in diff_values))