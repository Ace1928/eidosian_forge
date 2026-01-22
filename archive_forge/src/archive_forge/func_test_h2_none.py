from ase.build import molecule
def test_h2_none(cp2k_factory):
    calc = cp2k_factory.calc(basis_set=None, basis_set_file=None, max_scf=None, cutoff=None, force_eval_method=None, potential_file=None, poisson_solver=None, pseudo_potential=None, stress_tensor=False, xc=None, label='test_H2_inp', inp=inp)
    h2 = molecule('H2', calculator=calc)
    h2.center(vacuum=2.0)
    energy = h2.get_potential_energy()
    energy_ref = -30.6989595886
    diff = abs((energy - energy_ref) / energy_ref)
    assert diff < 1e-10