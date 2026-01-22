import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def unused_rules(self):
    unused_prod = []
    for s, v in self.Nonterminals.items():
        if not v:
            p = self.Prodnames[s][0]
            unused_prod.append(p)
    return unused_prod