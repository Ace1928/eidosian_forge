import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def undefined_symbols(self):
    result = []
    for p in self.Productions:
        if not p:
            continue
        for s in p.prod:
            if s not in self.Prodnames and s not in self.Terminals and (s != 'error'):
                result.append((s, p))
    return result