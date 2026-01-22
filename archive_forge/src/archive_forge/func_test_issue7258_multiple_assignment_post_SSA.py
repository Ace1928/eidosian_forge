import sys
import copy
import logging
import numpy as np
from numba import njit, jit, types
from numba.core import errors, ir
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.compiler import DefaultPassBuilder, CompilerBase
from numba.core.untyped_passes import ReconstructSSA, PreserveIR
from numba.core.typed_passes import NativeLowering
from numba.extending import overload
from numba.tests.support import MemoryLeakMixin, TestCase, override_config
def test_issue7258_multiple_assignment_post_SSA(self):
    cloned = []

    @register_pass(analysis_only=False, mutates_CFG=True)
    class CloneFoobarAssignments(FunctionPass):
        _name = 'clone_foobar_assignments_pass'

        def __init__(self):
            FunctionPass.__init__(self)

        def run_pass(self, state):
            mutated = False
            for blk in state.func_ir.blocks.values():
                to_clone = []
                for assign in blk.find_insts(ir.Assign):
                    if assign.target.name == 'foobar':
                        to_clone.append(assign)
                for assign in to_clone:
                    clone = copy.deepcopy(assign)
                    blk.insert_after(clone, assign)
                    mutated = True
                    cloned.append(clone)
            return mutated

    class CustomCompiler(CompilerBase):

        def define_pipelines(self):
            pm = DefaultPassBuilder.define_nopython_pipeline(self.state, 'custom_pipeline')
            pm._finalized = False
            pm.add_pass_after(CloneFoobarAssignments, ReconstructSSA)
            pm.add_pass_after(PreserveIR, NativeLowering)
            pm.finalize()
            return [pm]

    @njit(pipeline_class=CustomCompiler)
    def udt(arr):
        foobar = arr + 1
        return foobar
    arr = np.arange(10)
    self.assertPreciseEqual(udt(arr), arr + 1)
    self.assertEqual(len(cloned), 1)
    self.assertEqual(cloned[0].target.name, 'foobar')
    nir = udt.overloads[udt.signatures[0]].metadata['preserved_ir']
    self.assertEqual(len(nir.blocks), 1, 'only one block')
    [blk] = nir.blocks.values()
    assigns = blk.find_insts(ir.Assign)
    foobar_assigns = [stmt for stmt in assigns if stmt.target.name == 'foobar']
    self.assertEqual(len(foobar_assigns), 2, "expected two assignment statements into 'foobar'")
    self.assertEqual(foobar_assigns[0], foobar_assigns[1], 'expected the two assignment statements to be the same')