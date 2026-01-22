import abc
from typing import Any, List, Tuple
import scipy.sparse as sp
import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.utilities as u
from cvxpy.atoms.atom import Atom
from cvxpy.cvxcore.python import canonInterface
from cvxpy.expressions.constants import Constant
from cvxpy.utilities import performance_utils as perf
def is_pwl(self) -> bool:
    return all((arg.is_pwl() for arg in self.args))