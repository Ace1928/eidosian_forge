import numpy as np
from ase.io import read
from numpy.linalg import norm
def test_parse_socketio(testdir):
    write_output_socketio()
    traj = read('aims.out', ':', format='aims-output')
    a1, a2 = (traj[0], traj[1])
    f1, f2 = (a1.get_forces(), a2.get_forces())
    s1, s2 = (a1.get_stress(voigt=False), a2.get_stress(voigt=False))
    assert np.allclose(a1.positions[1, 0], 2.11313574)
    assert np.allclose(a2.positions[1, 0], 2.11313574)
    assert np.allclose(f1[0, 0], -1.08555415821635e-08)
    assert np.allclose(f2[1, 1], 0.000167235616064691)
    assert np.allclose(s1[0, 0], 6.913e-05)
    assert np.allclose(s2[0, 0], -0.0003266)