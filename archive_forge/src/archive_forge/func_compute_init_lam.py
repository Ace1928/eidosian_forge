import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.sparse import BlockMatrix, BlockVector
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve
import numpy as np
def compute_init_lam(nlp, x=None, lam_max=1000.0):
    if x is None:
        x = nlp.init_primals()
    else:
        assert x.size == nlp.n_primals()
    nlp.set_primals(x)
    assert nlp.n_ineq_constraints() == 0, 'only supported for equality constrained nlps for now'
    nx = nlp.n_primals()
    nc = nlp.n_constraints()
    jac = nlp.evaluate_jacobian()
    df = nlp.evaluate_grad_objective()
    kkt = BlockMatrix(2, 2)
    kkt.set_block(0, 0, identity(nx))
    kkt.set_block(1, 0, jac)
    kkt.set_block(0, 1, jac.transpose())
    zeros = np.zeros(nc)
    rhs = BlockVector(2)
    rhs.set_block(0, -df)
    rhs.set_block(1, zeros)
    flat_kkt = kkt.tocoo().tocsc()
    flat_rhs = rhs.flatten()
    sol = spsolve(flat_kkt, flat_rhs)
    return sol[nlp.n_primals():nlp.n_primals() + nlp.n_constraints()]