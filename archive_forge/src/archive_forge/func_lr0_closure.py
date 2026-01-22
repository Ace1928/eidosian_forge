import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def lr0_closure(self, I):
    self._add_count += 1
    J = I[:]
    didadd = True
    while didadd:
        didadd = False
        for j in J:
            for x in j.lr_after:
                if getattr(x, 'lr0_added', 0) == self._add_count:
                    continue
                J.append(x.lr_next)
                x.lr0_added = self._add_count
                didadd = True
    return J