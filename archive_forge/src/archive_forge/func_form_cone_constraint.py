import cvxpy as cp
from cvxpy import Variable
from cvxpy.constraints.constraint import Constraint
from cvxpy.constraints.exponential import ExpCone
from cvxpy.constraints.nonpos import NonNeg
from cvxpy.constraints.power import PowCone3D
from cvxpy.constraints.psd import PSD
from cvxpy.constraints.second_order import SOC
from cvxpy.constraints.zero import Zero
def form_cone_constraint(z: Variable, constraint: Constraint) -> Constraint:
    """
    Given a constraint represented as Ax+b in K for K a cvxpy cone, return an
    instantiated cvxpy constraint.
    """
    if isinstance(constraint, SOC):
        return SOC(t=z[0], X=z[1:])
    elif isinstance(constraint, NonNeg):
        return NonNeg(z)
    elif isinstance(constraint, ExpCone):
        n = z.shape[0]
        assert len(z.shape) == 1
        assert n % 3 == 0
        step = n // 3
        return ExpCone(z[:step], z[step:-step], z[-step:])
    elif isinstance(constraint, Zero):
        return Zero(z)
    elif isinstance(constraint, PSD):
        assert len(z.shape) == 1
        N = z.shape[0]
        n = int(N ** 0.5)
        assert N == n ** 2, 'argument is not a vectorized square matrix'
        z_mat = cp.reshape(z, (n, n))
        return PSD(z_mat)
    elif isinstance(constraint, PowCone3D):
        raise NotImplementedError
    else:
        raise NotImplementedError