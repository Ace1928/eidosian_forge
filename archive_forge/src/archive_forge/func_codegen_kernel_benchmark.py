from __future__ import annotations
import collections
import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import operator
import os
import textwrap
from typing import Any, Counter, Dict, Iterable, List, Optional, Set, Tuple, Union
import sympy
import torch
import torch._logging
from torch._prims_common import is_integer_dtype
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
from torch.utils._sympy.value_ranges import ValueRanges
from ..._dynamo.utils import counters
from .. import config, ir, scheduler
from ..codecache import code_hash, get_path, PyCodeCache
from ..dependencies import MemoryDep, StarDep
from ..ir import IRNode, ReductionHint, TritonTemplateBuffer
from ..optimize_indexing import indexing_dtype_strength_reduction
from ..scheduler import BaseScheduling, WhyNoFuse
from ..triton_heuristics import AutotuneHint
from ..utils import (
from ..virtualized import ops, V
from ..wrapper_benchmark import get_kernel_category_by_source_code
from .common import (
from .triton_utils import config_of, signature_of, signature_to_meta
def codegen_kernel_benchmark(self):
    result = IndentedBuffer()
    argdefs, call_args, signature = self.args.python_argdefs()
    result.writelines(['', '', 'def get_args():'])
    with result.indent():
        name_cnt = itertools.count()
        var_names = []
        for arg_name, arg_sig in zip(call_args, signature):
            var_name = f'arg_{next(name_cnt)}'
            buf = V.graph.get_buffer(arg_name)
            if buf:
                result.writeline(f"{var_name} = rand_strided({V.graph.sizevars.size_hints(buf.get_size())}, {V.graph.sizevars.size_hints(buf.get_stride())}, device='{buf.get_device()}', dtype={buf.get_dtype()})")
            elif arg_name in V.graph.constants:
                const_tensor = V.graph.constants[arg_name]
                result.writeline(f"{var_name} = rand_strided({V.graph.sizevars.size_hints(const_tensor.size())}, {V.graph.sizevars.size_hints(const_tensor.stride())}, device='{const_tensor.device}', dtype={const_tensor.dtype})")
            elif isinstance(arg_sig, SizeArg):
                symval_hint = V.graph.sizevars.size_hint(arg_sig.expr)
                if 'seed_offset' in arg_sig.name:
                    symval_hint = 0
                result.writeline(f'{var_name} = {symval_hint}')
            else:
                raise KeyError(f"Don't find the buffer or const tensor for {arg_name}")
            var_names.append(var_name)
        result.writeline(f'return {', '.join(var_names)},')
    result.writelines(['\n', '\n', 'def call(args):'])
    grid = []
    extra_args = []
    extra_args_str = None
    index = V.graph.scheduler.current_device.index
    with result.indent():
        result.writeline(f'with torch.cuda._DeviceGuard({index}):')
        with result.indent():
            result.writeline(f'torch.cuda.set_device({index})')
            for tree in self.range_trees:
                expr = pexpr(V.graph.sizevars.size_hint(tree.numel))
                if tree.prefix != 'r' or self.inside_reduction:
                    extra_args.append(expr)
                if tree.prefix != 'r':
                    grid.append(expr)
            stream_name = f'stream{index}'
            result.writeline(f'{stream_name} = get_cuda_stream({index})')
            if self.need_numel_args():
                extra_args_str = ', '.join(map(str, extra_args)) + ', '
            else:
                extra_args_str = ''
            result.writeline(f'{str(Placeholder.KERNEL_NAME)}.run(*args, {extra_args_str}grid=grid({', '.join(grid)}), stream={stream_name})')
    result.writelines(['\n', '\n', 'def benchmark_all_configs(args):'])
    with result.indent():
        result.writeline(f'with torch.cuda._DeviceGuard({index}):')
        with result.indent():
            result.writeline(f'torch.cuda.set_device({index})')
            result.writeline(f'return {str(Placeholder.KERNEL_NAME)}.benchmark_all_configs(*args, {extra_args_str}grid=grid({', '.join(grid)}))')
    ninplace_args = len(unique(self.args.inplace_buffers.values()))
    result.writelines(['\n', '\n', "if __name__ == '__main__':"])
    with result.indent():
        result.writeline('from torch._inductor.utils import get_num_bytes')
        result.writeline('from triton.testing import do_bench')
        result.writeline('')
        result.writeline('args = get_args()')
        result.writeline('ms = do_bench(lambda: call(args), rep=40, fast_flush=True)')
        result.writeline(f'num_gb = get_num_bytes(*args, num_in_out_args={ninplace_args}) / 1e9')
        result.writeline('gb_per_s = num_gb / (ms / 1e3)')
        result.writeline('print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")')
    return result