import os
import py_compile
import marshal
import inspect
import re
import tokenize
from .command import Command
from . import pluginlib
def search_dir(self, dir):
    names = os.listdir(dir)
    names.sort()
    dirs = []
    for name in names:
        full = os.path.join(dir, name)
        if name in self.bad_names:
            continue
        if os.path.isdir(full):
            dirs.append(full)
            continue
        for t in self.add_types:
            if name.lower().endswith(t.lower()):
                self.search_text(full)
        if not name.endswith('.py'):
            continue
        self.search_file(full)
    for dir in dirs:
        self.search_dir(dir)