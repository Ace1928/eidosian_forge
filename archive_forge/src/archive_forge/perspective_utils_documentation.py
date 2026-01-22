import cvxpy as cp
from cvxpy import Variable
from cvxpy.constraints.constraint import Constraint
from cvxpy.constraints.exponential import ExpCone
from cvxpy.constraints.nonpos import NonNeg
from cvxpy.constraints.power import PowCone3D
from cvxpy.constraints.psd import PSD
from cvxpy.constraints.second_order import SOC
from cvxpy.constraints.zero import Zero

    Given a constraint represented as Ax+b in K for K a cvxpy cone, return an
    instantiated cvxpy constraint.
    