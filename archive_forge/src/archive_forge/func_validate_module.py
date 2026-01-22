import re
import sys
import types
import copy
import os
import inspect
def validate_module(self, module):
    try:
        lines, linen = inspect.getsourcelines(module)
    except IOError:
        return
    fre = re.compile('\\s*def\\s+(t_[a-zA-Z_0-9]*)\\(')
    sre = re.compile('\\s*(t_[a-zA-Z_0-9]*)\\s*=')
    counthash = {}
    linen += 1
    for line in lines:
        m = fre.match(line)
        if not m:
            m = sre.match(line)
        if m:
            name = m.group(1)
            prev = counthash.get(name)
            if not prev:
                counthash[name] = linen
            else:
                filename = inspect.getsourcefile(module)
                self.log.error('%s:%d: Rule %s redefined. Previously defined on line %d', filename, linen, name, prev)
                self.error = True
        linen += 1