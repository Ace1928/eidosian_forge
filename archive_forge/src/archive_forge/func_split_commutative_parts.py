from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.matrices.dense import Matrix
from sympy.printing.pretty.stringpict import prettyForm
from sympy.core.containers import Tuple
from sympy.utilities.iterables import is_sequence
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.matrixutils import (
def split_commutative_parts(e):
    """Split into commutative and non-commutative parts."""
    c_part, nc_part = e.args_cnc()
    c_part = list(c_part)
    return (c_part, nc_part)