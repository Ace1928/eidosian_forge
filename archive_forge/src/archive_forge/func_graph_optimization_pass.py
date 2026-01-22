import collections
import itertools
import logging
import operator
import tempfile
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import (
import torch
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.distributed._spmd.graph_utils import (
from torch.distributed._spmd.iter_graph_module import IterGraphModule
from torch.fx.passes.shape_prop import TensorMetadata
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_unflatten
def graph_optimization_pass(prerequisites: Iterable[Callable], apply_after: Iterable[Callable]) -> Callable:
    """Define the contract of a graph optimization pass.

    All the passes should be wrapped with this decorator.
    `prerequisites` is used to annotate the prerequisite passes of the this pass.
    `apply_after` means that this wrapped pass must be applied after the passes
    in `apply_after`. The difference between `prerequisites` and `apply_after`
    is that all the passes in `prerequisites` must be applied to the graph and
    must be applifed before the wrapped pass while the passes `apply_after` are
    optional. But if a pass in `apply_after` is applied to the graph, it has to
    be done before the wrapped pass.
    Optimizer pass developers are required to add these fields accordingly and
    users need to follow the restrictions to avoid the assert.

    Current design has one limitation: users can only apply the optimizations
    once.  In some cases, we may need to run multiple the same optimization
    multiple time, e.g., optimization passes -> profiling the result -> apply
    optimization passes with the profiling result again. This limitation will be
    addressed limitation in the future.

    Args:
        prerequisites (Iterable[Callable]): the list of string to the names of
            passes which are the prerequisites of this pass.
        apply_after (Iterable[Callable]): the list of string to the names of
            passes that can not be applied after the wrapped pass.
    """

    def inner(func: Callable) -> Callable:

        def make_key(func: Callable) -> str:
            return f'{func.__module__}.{func.__name__}'
        func_key = make_key(func)
        _prerequisite_sets[func_key] = {make_key(f) for f in prerequisites}
        for apply_after_pass in apply_after:
            _apply_before_sets[make_key(apply_after_pass)].add(func_key)

        @wraps(func)
        def pass_wrapper(gm: Union[fx.GraphModule, IterGraphModule], *args: Any, **kwargs: Any) -> None:
            begin = time.time()
            assert isinstance(gm, (fx.GraphModule, IterGraphModule)), 'The first argument of the pass must be either fx.GraphModule or IterGraphModule.'
            assert func_key not in _optimized_func, f'Cannot apply {func_key} twice.'
            invalid_passes = _apply_before_sets[func_key].intersection(_optimized_func)
            assert not invalid_passes, f'{invalid_passes} must be applied after {func_key}.'
            assert _prerequisite_sets[func_key].issubset(_optimized_func), f'{_prerequisite_sets[func_key] - _optimized_func} are the prerequisites of {func_key} but are not applified. Applied passes are {_optimized_func}.'
            func(gm, *args, **kwargs)
            gm.graph.lint()
            gm.graph.eliminate_dead_code()
            gm.recompile()
            _optimized_func.add(func_key)
            prefix = f'after_{func.__name__}'
            if _dump_graph_folder:
                if isinstance(gm, IterGraphModule):
                    dump_graphs_to_files({f'{prefix}_setup_gm': gm.setup_gm, f'{prefix}_main_gm': gm.main_gm, f'{prefix}_cleanup_gm': gm.cleanup_gm}, _dump_graph_folder)
                else:
                    dump_graphs_to_files({prefix: gm}, _dump_graph_folder)
            logger.info('Spent %f seconds applying %s', time.time() - begin, func_key)
        return pass_wrapper
    return inner