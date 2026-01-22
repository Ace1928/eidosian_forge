from sympy.utilities.iterables import kbins
def unify_var(var, x, s, **fns):
    if var in s:
        yield from unify(s[var], x, s, **fns)
    elif occur_check(var, x):
        pass
    elif isinstance(var, CondVariable) and var.valid(x):
        yield assoc(s, var, x)
    elif isinstance(var, Variable):
        yield assoc(s, var, x)