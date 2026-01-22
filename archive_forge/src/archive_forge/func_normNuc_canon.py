from typing import List, Tuple
from cvxpy.atoms.affine.bmat import bmat
from cvxpy.atoms.affine.trace import trace
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.variable import Variable
def normNuc_canon(expr, args) -> Tuple[float, List[Constraint]]:
    A = args[0]
    m, n = A.shape
    constraints = []
    U = Variable(shape=(m, m), symmetric=True)
    V = Variable(shape=(n, n), symmetric=True)
    X = bmat([[U, A], [A.T, V]])
    constraints.append(X >> 0)
    trace_value = 0.5 * (trace(U) + trace(V))
    return (trace_value, constraints)