import contextlib
import functools
from llvmlite.ir import instructions, types, values
def landingpad(self, typ, name='', cleanup=False):
    inst = instructions.LandingPadInstr(self.block, typ, name, cleanup)
    self._insert(inst)
    return inst