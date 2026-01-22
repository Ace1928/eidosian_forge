import itertools
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
from scipy import linalg
import scipy.linalg._decomp_update as _decomp_update
from scipy.linalg._decomp_update import qr_delete, qr_update, qr_insert
def test_form_qTu():
    q_order = ['F', 'C']
    q_shape = [(8, 8)]
    u_order = ['F', 'C', 'A']
    u_shape = [1, 3]
    dtype = ['f', 'd', 'F', 'D']
    for qo, qs, uo, us, d in itertools.product(q_order, q_shape, u_order, u_shape, dtype):
        if us == 1:
            check_form_qTu(qo, qs, uo, us, 1, d)
            check_form_qTu(qo, qs, uo, us, 2, d)
        else:
            check_form_qTu(qo, qs, uo, us, 2, d)