import os
from typing import (
from blib2to3.pgen2 import grammar, token, tokenize
from blib2to3.pgen2.tokenize import GoodTokenInfo
def parse_alt(self) -> Tuple['NFAState', 'NFAState']:
    a, b = self.parse_item()
    while self.value in ('(', '[') or self.type in (token.NAME, token.STRING):
        c, d = self.parse_item()
        b.addarc(c)
        b = d
    return (a, b)