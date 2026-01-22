import os
from typing import (
from blib2to3.pgen2 import grammar, token, tokenize
from blib2to3.pgen2.tokenize import GoodTokenInfo
def dump_dfa(self, name: str, dfa: Sequence['DFAState']) -> None:
    print('Dump of DFA for', name)
    for i, state in enumerate(dfa):
        print('  State', i, state.isfinal and '(final)' or '')
        for label, next in sorted(state.arcs.items()):
            print('    %s -> %d' % (label, dfa.index(next)))