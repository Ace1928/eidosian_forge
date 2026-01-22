import collections
import types as pytypes
import numpy as np
from numba.core.compiler import run_frontend, Flags, StateDict
from numba import jit, njit, literal_unroll
from numba.core import types, errors, ir, rewrites, ir_utils, utils, cpu
from numba.core import postproc
from numba.core.inline_closurecall import InlineClosureCallPass
from numba.tests.support import (TestCase, MemoryLeakMixin, SerialMixin,
from numba.core.analysis import dead_branch_prune, rewrite_semantic_constants
from numba.core.untyped_passes import (ReconstructSSA, TranslateByteCode,
from numba.core.compiler import DefaultPassBuilder, CompilerBase, PassManager
def assert_prune(self, func, args_tys, prune, *args, **kwargs):
    func_ir = compile_to_ir(func)
    before = func_ir.copy()
    if self._DEBUG:
        print('=' * 80)
        print('before inline')
        func_ir.dump()
    inline_pass = InlineClosureCallPass(func_ir, cpu.ParallelOptions(False))
    inline_pass.run()
    post_proc = postproc.PostProcessor(func_ir)
    post_proc.run()
    rewrite_semantic_constants(func_ir, args_tys)
    if self._DEBUG:
        print('=' * 80)
        print('before prune')
        func_ir.dump()
    dead_branch_prune(func_ir, args_tys)
    after = func_ir
    if self._DEBUG:
        print('after prune')
        func_ir.dump()
    before_branches = self.find_branches(before)
    self.assertEqual(len(before_branches), len(prune))
    expect_removed = []
    for idx, prune in enumerate(prune):
        branch = before_branches[idx]
        if prune is True:
            expect_removed.append(branch.truebr)
        elif prune is False:
            expect_removed.append(branch.falsebr)
        elif prune is None:
            pass
        elif prune == 'both':
            expect_removed.append(branch.falsebr)
            expect_removed.append(branch.truebr)
        else:
            assert 0, 'unreachable'
    original_labels = set([_ for _ in before.blocks.keys()])
    new_labels = set([_ for _ in after.blocks.keys()])
    try:
        self.assertEqual(new_labels, original_labels - set(expect_removed))
    except AssertionError as e:
        print('new_labels', sorted(new_labels))
        print('original_labels', sorted(original_labels))
        print('expect_removed', sorted(expect_removed))
        raise e
    supplied_flags = kwargs.pop('flags', {'nopython': True})
    cres = jit(args_tys, **supplied_flags)(func).overloads[args_tys]
    if args is None:
        res = cres.entry_point()
        expected = func()
    else:
        res = cres.entry_point(*args)
        expected = func(*args)
    self.assertEqual(res, expected)