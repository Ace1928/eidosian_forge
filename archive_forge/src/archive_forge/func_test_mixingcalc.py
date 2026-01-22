def test_mixingcalc():
    """This test checks the basic functionality of the MixingCalculators.
    The example system is based on the SinglePointCalculator test case.
    """
    import numpy as np
    from ase.build import fcc111
    from ase.calculators.emt import EMT
    from ase.calculators.mixing import SumCalculator, LinearCombinationCalculator, AverageCalculator, MixedCalculator
    from ase.constraints import FixAtoms
    atoms = fcc111('Cu', (2, 2, 1), vacuum=10.0)
    atoms[0].x += 0.2
    calc = EMT()
    atoms.calc = calc
    forces = atoms.get_forces()
    atoms1 = atoms.copy()
    calc1 = SumCalculator([EMT(), EMT()])
    atoms1.calc = calc1
    atoms2 = atoms.copy()
    SumCalculator(calcs=[EMT(), EMT()], atoms=atoms2)
    assert np.isclose(2 * forces, atoms1.get_forces()).all()
    assert np.isclose(2 * forces, atoms2.get_forces()).all()
    atoms1[0].x += 0.2
    assert not np.isclose(2 * forces, atoms1.get_forces()).all()
    atoms1.set_constraint(FixAtoms(indices=[atom.index for atom in atoms]))
    assert np.isclose(0, atoms1.get_forces()).all()
    atoms1 = atoms.copy()
    calc1 = AverageCalculator([EMT(), EMT()])
    atoms1.calc = calc1
    atoms2 = atoms.copy()
    LinearCombinationCalculator([EMT(), EMT()], weights=[0.5, 0.5], atoms=atoms2)
    assert np.isclose(forces, atoms1.get_forces()).all()
    assert np.isclose(forces, atoms2.get_forces()).all()
    atoms1[0].x += 0.2
    assert not np.isclose(2 * forces, atoms1.get_forces()).all()
    try:
        calc1 = LinearCombinationCalculator([], [])
    except ValueError:
        assert True
    try:
        calc1 = AverageCalculator([])
    except ValueError:
        assert True
    w1, w2 = (0.78, 0.22)
    atoms1 = atoms.copy()
    atoms1.calc = EMT()
    E_tot = atoms1.get_potential_energy()
    calc1 = MixedCalculator(EMT(), EMT(), w1, w2)
    E1, E2 = calc1.get_energy_contributions(atoms1)
    assert np.isclose(E1, E_tot)
    assert np.isclose(E2, E_tot)