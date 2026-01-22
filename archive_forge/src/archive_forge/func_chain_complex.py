from ... import sage_helper
from .. import t3mlite as t3m
def chain_complex(self):
    if self._chain_complex is None:
        self._chain_complex = ChainComplex({1: self.B1(), 2: self.B2()}, degree=-1)
    return self._chain_complex