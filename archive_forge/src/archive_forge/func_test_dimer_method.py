from ase.build import fcc100, add_adsorbate
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.dimer import DimerControl, MinModeAtoms, MinModeTranslate
def test_dimer_method(testdir):
    atoms = fcc100('Pt', size=(2, 2, 1), vacuum=10.0)
    add_adsorbate(atoms, 'Pt', 1.611, 'hollow')
    mask = [atom.tag > 0 for atom in atoms]
    atoms.set_constraint(FixAtoms(mask=mask))
    atoms.calc = EMT()
    atoms.get_potential_energy()
    with DimerControl(initial_eigenmode_method='displacement', displacement_method='vector', logfile=None, mask=[0, 0, 0, 0, 1]) as d_control:
        d_atoms = MinModeAtoms(atoms, d_control)
        displacement_vector = [[0.0] * 3] * 5
        displacement_vector[-1][1] = -0.1
        d_atoms.displace(displacement_vector=displacement_vector)
        with MinModeTranslate(d_atoms, trajectory='dimer_method.traj', logfile=None) as dim_rlx:
            dim_rlx.run(fmax=0.001)
    tolerance = 0.001
    assert d_atoms.get_barrier_energy() - 1.03733136918 < tolerance
    assert abs(d_atoms.get_curvature() + 0.900467048707) < tolerance
    assert d_atoms.get_eigenmode()[-1][1] < -0.99
    assert abs(d_atoms.get_positions()[-1][1]) < tolerance