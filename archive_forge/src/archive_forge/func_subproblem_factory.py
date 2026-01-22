from ._trlib import TRLIBQuadraticSubproblem
def subproblem_factory(x, fun, jac, hess, hessp):
    return TRLIBQuadraticSubproblem(x, fun, jac, hess, hessp, tol_rel_i=tol_rel_i, tol_rel_b=tol_rel_b, disp=disp)