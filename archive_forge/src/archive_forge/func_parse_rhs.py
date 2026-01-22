import os
from typing import (
from blib2to3.pgen2 import grammar, token, tokenize
from blib2to3.pgen2.tokenize import GoodTokenInfo
def parse_rhs(self) -> Tuple['NFAState', 'NFAState']:
    a, z = self.parse_alt()
    if self.value != '|':
        return (a, z)
    else:
        aa = NFAState()
        zz = NFAState()
        aa.addarc(a)
        z.addarc(zz)
        while self.value == '|':
            self.gettoken()
            a, z = self.parse_alt()
            aa.addarc(a)
            z.addarc(zz)
        return (aa, zz)