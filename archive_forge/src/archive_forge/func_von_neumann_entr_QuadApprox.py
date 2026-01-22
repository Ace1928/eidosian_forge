from typing import List, Tuple
import numpy as np
import cvxpy as cp
from cvxpy.atoms.affine.upper_tri import upper_tri
from cvxpy.constraints.constraint import Constraint
from cvxpy.constraints.exponential import (
from cvxpy.constraints.zero import Zero
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.dcp2cone.canonicalizers.von_neumann_entr_canon import (
def von_neumann_entr_QuadApprox(expr, args):
    m, k = (expr.quad_approx[0], expr.quad_approx[1])
    epi, initial_cons = von_neumann_entr_canon(expr, args)
    cons = []
    for con in initial_cons:
        if isinstance(con, ExpCone):
            qa_con = con.as_quad_approx(m, k)
            qa_con_canon_lead, qa_con_canon = RelEntrConeQuad_canon(qa_con, None)
            cons.append(qa_con_canon_lead)
            cons.extend(qa_con_canon)
        else:
            cons.append(con)
    return (epi, cons)