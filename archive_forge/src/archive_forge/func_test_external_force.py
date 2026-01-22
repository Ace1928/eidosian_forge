from ase import Atoms
from ase.constraints import ExternalForce, FixBondLength
from ase.optimize import FIRE
from ase.calculators.emt import EMT
from numpy.linalg import norm
def test_external_force():
    """Tests for class ExternalForce in ase/constraints.py"""
    f_ext = 0.2
    atom1 = 0
    atom2 = 1
    atom3 = 2
    atoms = Atoms('H3', positions=[(0, 0, 0), (0.751, 0, 0), (0, 1.0, 0)])
    atoms.calc = EMT()
    optimize(atoms)
    dist1 = atoms.get_distance(atom1, atom2)
    con1 = ExternalForce(atom1, atom2, f_ext)
    atoms.set_constraint(con1)
    optimize(atoms)
    dist2 = atoms.get_distance(atom1, atom2)
    assert dist2 > dist1
    con2 = FixBondLength(atom1, atom2)
    atoms.set_constraint([con1, con2])
    optimize(atoms)
    f_con = con2.constraint_forces
    assert norm(f_con[0]) <= fmax
    atoms.set_constraint(con2)
    optimize(atoms)
    f_con = con2.constraint_forces[0]
    assert round(norm(f_con), 2) == round(abs(f_ext), 2)
    f_ext *= 2
    con1 = ExternalForce(atom1, atom2, f_ext)
    d1 = atoms.get_distance(atom1, atom3)
    con2 = FixBondLength(atom1, atom3)
    atoms.set_constraint([con1, con2])
    optimize(atoms)
    d2 = atoms.get_distance(atom1, atom3)
    assert round(d1, 5) == round(d2, 5)