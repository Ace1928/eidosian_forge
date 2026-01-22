from cvxpy.expressions.variable import Variable
def find_non_prop_leaves(expr, res=None):
    if res is None:
        res = []
    if len(expr.args) == 0 and getattr(expr, prop_name)():
        return res
    if not getattr(expr, prop_name)() and all((getattr(child, prop_name)() for child in expr.args)):
        str_expr = str(expr)
        if discipline_type == DGP and isinstance(expr, Variable):
            str_expr += ' <-- needs to be declared positive'
        res.append(str_expr)
    for child in expr.args:
        res = find_non_prop_leaves(child, res)
    return res