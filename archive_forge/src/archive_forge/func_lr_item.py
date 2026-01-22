import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def lr_item(self, n):
    if n > len(self.prod):
        return None
    p = LRItem(self, n)
    try:
        p.lr_after = Prodnames[p.prod[n + 1]]
    except (IndexError, KeyError):
        p.lr_after = []
    try:
        p.lr_before = p.prod[n - 1]
    except IndexError:
        p.lr_before = None
    return p