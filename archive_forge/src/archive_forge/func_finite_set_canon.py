from cvxpy.constraints.finite_set import FiniteSet
def finite_set_canon(expr, args):
    ineq_form, id = expr.get_data()
    return (FiniteSet(args[0], args[1], ineq_form, id), [])