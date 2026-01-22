from __future__ import annotations
from dask.rewrite import VAR, RewriteRule, RuleSet, Traverser, args, head
from dask.utils_test import add, inc
def repl_list(sd):
    x = sd['x']
    if isinstance(x, list):
        return x
    else:
        return (list, x)