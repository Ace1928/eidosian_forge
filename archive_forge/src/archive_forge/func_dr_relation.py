import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def dr_relation(self, C, trans, nullable):
    dr_set = {}
    state, N = trans
    terms = []
    g = self.lr0_goto(C[state], N)
    for p in g:
        if p.lr_index < p.len - 1:
            a = p.prod[p.lr_index + 1]
            if a in self.grammar.Terminals:
                if a not in terms:
                    terms.append(a)
    if state == 0 and N == self.grammar.Productions[0].prod[0]:
        terms.append('$end')
    return terms