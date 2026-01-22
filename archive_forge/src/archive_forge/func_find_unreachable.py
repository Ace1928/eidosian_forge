import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def find_unreachable(self):

    def mark_reachable_from(s):
        if s in reachable:
            return
        reachable.add(s)
        for p in self.Prodnames.get(s, []):
            for r in p.prod:
                mark_reachable_from(r)
    reachable = set()
    mark_reachable_from(self.Productions[0].prod[0])
    return [s for s in self.Nonterminals if s not in reachable]