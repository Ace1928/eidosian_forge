from collections import defaultdict, OrderedDict
from contextlib import contextmanager
import sys
import gast as ast
def defs(self, node):
    name = node.id
    stars = []
    for d in reversed(self._definitions):
        if name in d:
            return d[name] if not stars else stars + list(d[name])
        if '*' in d:
            stars.extend(d['*'])
    d = self.chains.setdefault(node, Def(node))
    if self._undefs:
        self._undefs[-1][name].append((d, stars))
    if stars:
        return stars + [d]
    else:
        if not self._undefs:
            self.unbound_identifier(name, node)
        return [d]