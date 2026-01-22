from numba import cuda, float32, int32
from numba.core.errors import NumbaInvalidConfigWarning
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba.tests.support import ignore_internal_warnings
import re
import unittest
import warnings
def test_lineinfo_in_device_function(self):

    @cuda.jit(lineinfo=True)
    def callee(x):
        x[0] += 1

    @cuda.jit(lineinfo=True)
    def caller(x):
        x[0] = 1
        callee(x)
    sig = (int32[:],)
    self._check(caller, sig=sig, expect=True)
    ptx = caller.inspect_asm(sig)
    ptxlines = ptx.splitlines()
    devfn_start = re.compile('^\\.weak\\s+\\.func')
    for line in ptxlines:
        if devfn_start.match(line) is not None:
            self.fail(f'Found device function in PTX:\n\n{ptx}')
    loc_directive = self._loc_directive_regex()
    found = False
    for line in ptxlines:
        if loc_directive.search(line) is not None:
            if 'inlined_at' in line:
                found = True
                break
    if not found:
        self.fail(f'No .loc directive with inlined_at info foundin:\n\n{ptx}')
    llvm = caller.inspect_llvm(sig)
    subprograms = 0
    for line in llvm.splitlines():
        if 'distinct !DISubprogram' in line:
            subprograms += 1
    expected_subprograms = 3
    self.assertEqual(subprograms, expected_subprograms, f'"Expected {expected_subprograms} DISubprograms; got {subprograms}')