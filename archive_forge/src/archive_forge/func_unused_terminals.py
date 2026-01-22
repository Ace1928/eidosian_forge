import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def unused_terminals(self):
    unused_tok = []
    for s, v in self.Terminals.items():
        if s != 'error' and (not v):
            unused_tok.append(s)
    return unused_tok