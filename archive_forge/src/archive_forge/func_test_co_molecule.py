from ase.md.analysis import DiffusionCoefficient
from ase.atoms import Atoms
from ase.units import fs as fs_conversion
def test_co_molecule():
    co = Atoms('CO', positions=[(0, 0, 0), (0, 0, 1)])
    traj_co = [co.copy() for i in range(2)]
    traj_co[1].set_positions([(-1, -1, -1), (-1, -1, 0)])
    dc_co = DiffusionCoefficient(traj_co, timestep, molecule=False)
    dc_co.calculate(ignore_n_images=0, number_of_segments=1)
    ans = dc_co.get_diffusion_coefficients()[0][0]
    assert abs(ans - ans_orig) < eps
    for index in range(2):
        dc_co = DiffusionCoefficient(traj_co, timestep, atom_indices=[index], molecule=False)
        dc_co.calculate()
        ans = dc_co.get_diffusion_coefficients()[0][0]
        assert abs(ans - ans_orig) < eps
    dc_co = DiffusionCoefficient(traj_co, timestep, molecule=True)
    dc_co.calculate(ignore_n_images=0, number_of_segments=1)
    ans = dc_co.get_diffusion_coefficients()[0][0]
    assert abs(ans - ans_orig) < eps