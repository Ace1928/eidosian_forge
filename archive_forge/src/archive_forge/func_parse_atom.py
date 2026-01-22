import os
from typing import (
from blib2to3.pgen2 import grammar, token, tokenize
from blib2to3.pgen2.tokenize import GoodTokenInfo
def parse_atom(self) -> Tuple['NFAState', 'NFAState']:
    if self.value == '(':
        self.gettoken()
        a, z = self.parse_rhs()
        self.expect(token.OP, ')')
        return (a, z)
    elif self.type in (token.NAME, token.STRING):
        a = NFAState()
        z = NFAState()
        a.addarc(z, self.value)
        self.gettoken()
        return (a, z)
    else:
        self.raise_error('expected (...) or NAME or STRING, got %s/%s', self.type, self.value)
        raise AssertionError