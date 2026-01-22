import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def lr0_items(self):
    C = [self.lr0_closure([self.grammar.Productions[0].lr_next])]
    i = 0
    for I in C:
        self.lr0_cidhash[id(I)] = i
        i += 1
    i = 0
    while i < len(C):
        I = C[i]
        i += 1
        asyms = {}
        for ii in I:
            for s in ii.usyms:
                asyms[s] = None
        for x in asyms:
            g = self.lr0_goto(I, x)
            if not g or id(g) in self.lr0_cidhash:
                continue
            self.lr0_cidhash[id(g)] = len(C)
            C.append(g)
    return C