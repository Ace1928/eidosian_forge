import os
from typing import (
from blib2to3.pgen2 import grammar, token, tokenize
from blib2to3.pgen2.tokenize import GoodTokenInfo
def make_first(self, c: PgenGrammar, name: str) -> Dict[int, int]:
    rawfirst = self.first[name]
    assert rawfirst is not None
    first = {}
    for label in sorted(rawfirst):
        ilabel = self.make_label(c, label)
        first[ilabel] = 1
    return first