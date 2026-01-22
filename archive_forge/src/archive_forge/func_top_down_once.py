from sympy.strategies.util import basic_fns
from sympy.strategies.core import chain, do_one
def top_down_once(rule, fns=basic_fns):
    """Apply a rule down a tree - stop on success."""
    return do_one(rule, lambda expr: sall(top_down(rule, fns), fns)(expr))