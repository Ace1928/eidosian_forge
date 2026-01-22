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
def codegen_kernel(self, name=None):
    from triton import next_power_of_2
    code = IndentedBuffer()
    size_hints = []
    for numel in self.numels:
        numel_hint = V.graph.sizevars.symbolic_hint(numel)
        if not isinstance(numel_hint, (int, sympy.Integer)):
            size_hint = 8192
        else:
            size_hint = next_power_of_2(int(numel_hint))
        size_hints.append(size_hint)
    if self.persistent_reduction:
        assert self.inside_reduction
        heuristics = 'persistent_reduction'
    elif self.inside_reduction:
        heuristics = 'reduction'
    else:
        size_hints.pop()
        heuristics = 'pointwise'
    if name is None:
        code.splice(f'\n                    import triton\n                    import triton.language as tl\n                    from torch._inductor.ir import ReductionHint\n                    from torch._inductor.ir import TileHint\n                    from torch._inductor.triton_heuristics import AutotuneHint, {heuristics}\n                    from torch._inductor.utils import instance_descriptor\n                    from torch._inductor import triton_helpers\n                ')
        if config.benchmark_kernel:
            code.splice(self.imports_for_benchmark_kernel())
    argdefs, _, signature = self.args.python_argdefs()
    for i, arg in enumerate(signature):
        if isinstance(arg, SizeArg) and arg.expr in V.graph.sizevars.inv_precomputed_replacements:
            signature[i] = SizeArg(arg.name, V.graph.sizevars.inv_precomputed_replacements[arg.expr])
    mutated_args = set()
    for mutation in self.mutations:
        if mutation in self.args.input_buffers:
            mutated_args.add(self.args.input_buffers[mutation])
        if mutation in self.args.inplace_buffers and mutation not in V.graph.removed_buffers and (mutation not in self.removed_buffers):
            mutated_args.add(self.args.inplace_buffers[mutation].inner_name)
        if mutation in self.args.output_buffers:
            mutated_args.add(self.args.output_buffers[mutation])
    mutated_args = sorted(mutated_args)
    triton_meta_signature = signature_to_meta(signature, size_dtype=self.index_dtype)
    triton_meta = {'signature': triton_meta_signature, 'device': V.graph.scheduler.current_device.index, 'device_type': V.graph.scheduler.current_device.type, 'constants': {}}
    inductor_meta = {'autotune_hints': set(self.autotune_hints), 'kernel_name': str(Placeholder.DESCRIPTIVE_NAME), 'mutated_arg_names': mutated_args}
    for tree in self.range_trees:
        if tree.prefix != 'r' or self.inside_reduction:
            sizearg = SizeArg(f'{tree.prefix}numel', tree.numel)
            signature.append(sizearg)
            triton_meta_signature[len(argdefs)] = signature_of(sizearg, size_dtype=self.index_dtype)
            argdefs.append(f'{tree.prefix}numel')
    triton_meta['configs'] = [config_of(signature)]
    for tree in self.range_trees:
        if tree.prefix == 'r' and (not self.inside_reduction or self.persistent_reduction):
            continue
        if tree.prefix == 'x' and self.no_x_dim:
            continue
        argdefs.append(f'{tree.prefix.upper()}BLOCK : tl.constexpr')
    if self.inside_reduction:
        reduction_hint = self.reduction_hint
        heuristics_line = f'\n                @{heuristics}(\n                    size_hints={size_hints!r},\n                    reduction_hint={reduction_hint},\n                    filename=__file__,\n                    triton_meta={triton_meta!r},\n                    inductor_meta={inductor_meta!r}\n                )\n                @triton.jit\n            '
    else:
        tile_hint = ''
        if len(size_hints) == 2:
            if len(signature) == 4:
                tile_hint = 'tile_hint=TileHint.SQUARE,'
            else:
                tile_hint = 'tile_hint=TileHint.DEFAULT,'
        heuristics_line = f'\n                @{heuristics}(\n                    size_hints={size_hints!r}, {tile_hint}\n                    filename=__file__,\n                    triton_meta={triton_meta!r},\n                    inductor_meta={inductor_meta!r},\n                    min_elem_per_thread={self.min_elem_per_thread}\n                )\n                @triton.jit\n            '
    code.splice(heuristics_line)
    code.writeline(f'def {name or str(Placeholder.KERNEL_NAME)}({', '.join(argdefs)}):')
    self.codegen_body()
    with code.indent():
        self.codegen_static_numels(code)
        for old, new in self.args.aliases():
            code.writeline(f'{old} = {new}')
        code.splice(self.body)
    if config.benchmark_kernel:
        code.splice(self.codegen_kernel_benchmark())
    return code.getvalue()