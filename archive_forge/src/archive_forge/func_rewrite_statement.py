import operator
from functools import reduce
from collections import namedtuple, defaultdict
from .controlflow import CFGraph
from numba.core import types, errors, ir, consts
from numba.misc import special
def rewrite_statement(func_ir, stmt, new_val):
    """
        Rewrites the stmt as a ir.Const new_val and fixes up the entries in
        func_ir._definitions
        """
    stmt.value = ir.Const(new_val, stmt.loc)
    defns = func_ir._definitions[stmt.target.name]
    repl_idx = defns.index(val)
    defns[repl_idx] = stmt.value