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
def repro_analyze(options, mod, load_args):
    from torch._inductor.compile_fx import compile_fx_inner
    from torch._inductor.hooks import intermediate_hook
    mod, args = repro_common(options, mod, load_args)
    with tqdm(desc='Compiling'):
        compiled = compile_fx_inner(mod, args)
    total = counters['inductor']['intermediate_hooks']
    known_names = set()

    def save_hook(name, val):
        known_names.add(name)
        if not options.skip_saving_inductor_intermediates:
            writer.write_tensor(os.path.join('inductor', name), val)
        pbar.update(1)
    writer = torch.utils._content_store.ContentStoreWriter(options.save_dir, stable_hash=options.stable_hash)
    reader = torch.utils._content_store.ContentStoreReader(options.save_dir)
    new_args = clone_inputs(args)
    with intermediate_hook(save_hook), tqdm(desc='Saving inductor intermediates', total=total) as pbar:
        compiled(new_args)
        assert not new_args

    def compare_tuples(tuple1, tuple2):
        diff_indices = [i for i in range(len(tuple1)) if tuple1[i] != tuple2[i]]
        diff_values = [(tuple1[i], tuple2[i]) for i in diff_indices]
        if not diff_values:
            return None
        else:
            return ' and '.join((f'{a} != {b}' for a, b in diff_values))

    def check_hook(name, val):
        meta = writer.compute_tensor_metadata(val)
        meta2 = reader.read_tensor_metadata(os.path.join('inductor', name))
        reason = compare_tuples(meta, meta2)
        if reason is not None:
            pbar.write(f'NONDETERMINISTIC INDUCTOR at {name} ({reason})')
        pbar.update(1)
    if not options.skip_check_deterministic:
        new_args = clone_inputs(args)
        with intermediate_hook(check_hook), tqdm(desc='Checking inductor determinism', total=total) as pbar:
            compiled(new_args)
            assert not new_args

    class WriterInterp(fx.Interpreter):

        def __init__(self, mod, subdir):
            super().__init__(mod)
            self.subdir = subdir

        def run_node(self, n):
            r = super().run_node(n)
            name = n.name
            if name in known_names:
                pbar.update(1)
                writer.write_tensor(os.path.join(self.subdir, name), r)
            return r
    if not options.skip_saving_float64_intermediates:
        new_mod, new_args = cast_to_fp64(copy.deepcopy(mod), clone_inputs(args))
        with tqdm(desc='Saving float64 intermediates', total=total) as pbar:
            WriterInterp(new_mod, 'float64').boxed_run(new_args)
        assert not new_args

    class ExactReaderInterp(fx.Interpreter):

        def run_node(self, n):
            r = super().run_node(n)
            name = n.name
            if name in known_names:
                meta = writer.compute_tensor_metadata(r)
                meta2 = reader.read_tensor_metadata(os.path.join('float64', name))
                reason = compare_tuples(meta, meta2)
                if reason is not None:
                    pbar.write(f'NONDETERMINISTIC FLOAT64 at {name} ({reason})')
                pbar.update(1)
            return r
    if not options.skip_check_deterministic:
        new_mod, new_args = cast_to_fp64(copy.deepcopy(mod), clone_inputs(args))
        with tqdm(desc='Checking float64 determinism', total=total) as pbar:
            ExactReaderInterp(new_mod).boxed_run(new_args)
            assert not new_args

    class ReaderInterp(fx.Interpreter):

        def run_node(self, n):
            r = super().run_node(n)
            name = n.name
            if name in known_names:
                inductor = reader.read_tensor(os.path.join('inductor', name))
                float64 = reader.read_tensor(os.path.join('float64', name))
                logged = False

                def log_error(msg, *args):
                    nonlocal logged
                    logged = True
                    pbar.write(f'DIVERGED at {name}: {msg % args}')
                if not same(r, inductor, float64, tol=torch._dynamo.config.repro_tolerance, equal_nan=True, log_error=log_error):
                    assert logged
                pbar.update(1)
            return r
    with tqdm(desc='Checking divergence', total=total) as pbar:
        ReaderInterp(mod).boxed_run(args)
    assert not args