import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
def get_implies(name, _caller=set()):
    implies = set()
    d = self.feature_supported[name]
    for i in d.get('implies', []):
        implies.add(i)
        if i in _caller:
            continue
        _caller.add(name)
        implies = implies.union(get_implies(i, _caller))
    return implies