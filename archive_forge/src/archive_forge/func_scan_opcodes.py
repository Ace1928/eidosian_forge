import dis
import importlib._bootstrap_external
import importlib.machinery
import marshal
import os
import io
import sys
def scan_opcodes(self, co):
    for name in dis._find_store_names(co):
        yield ('store', (name,))
    for name, level, fromlist in dis._find_imports(co):
        if level == 0:
            yield ('absolute_import', (fromlist, name))
        else:
            yield ('relative_import', (level, fromlist, name))