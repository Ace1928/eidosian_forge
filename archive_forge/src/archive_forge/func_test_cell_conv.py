import numpy as np
from ase.geometry import cell_to_cellpar as c2p, cellpar_to_cell as p2c
def test_cell_conv():
    assert (p2c([1, 1, 1, 90, 90, 90]) == np.eye(3)).all()
    a = 5.43
    d = a / 2.0
    h = a / np.sqrt(2.0)
    si_prim_p = np.array([h] * 3 + [60.0] * 3)
    si_prim_m = np.array([[0.0, d, d], [d, 0.0, d], [d, d, 0.0]])
    si_prim_m2 = np.array([[2.0, 0.0, 0.0], [1.0, np.sqrt(3.0), 0.0], [1.0, np.sqrt(3.0) / 3.0, 2 * np.sqrt(2 / 3)]])
    si_prim_m2 *= h / 2.0
    si_ortho_p = np.array([h] * 2 + [a] + [90.0] * 3)
    si_ortho_m = np.array([[h, 0.0, 0.0], [0.0, h, 0.0], [0.0, 0.0, a]])
    si_cubic_p = np.array([a] * 3 + [90.0] * 3)
    si_cubic_m = np.array([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]])
    assert_equal(c2p(si_prim_m), si_prim_p)
    assert_equal(c2p(si_prim_m2), si_prim_p)
    assert_equal(c2p(si_ortho_m), si_ortho_p)
    assert_equal(c2p(si_cubic_m), si_cubic_p)
    assert not nearly_equal(c2p(si_prim_m), si_ortho_p)
    assert_equal(p2c(si_prim_p), si_prim_m2)
    assert_equal(p2c(si_ortho_p), si_ortho_m)
    assert_equal(p2c(si_cubic_p), si_cubic_m)
    assert not nearly_equal(p2c(si_prim_p), si_ortho_m)
    ref1 = si_prim_m2[:]
    ref2 = si_ortho_m[:]
    ref3 = si_cubic_m[:]
    for i in range(20):
        ref1[:] = p2c(c2p(ref1))
        ref2[:] = p2c(c2p(ref2))
        ref3[:] = p2c(c2p(ref3))
    assert_equal(ref1, si_prim_m2)
    assert_equal(ref2, si_ortho_m)
    assert_equal(ref3, si_cubic_m)