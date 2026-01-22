from math import radians, sin, cos
import pytest
from ase import Atoms
from ase.neb import NEB
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.optimize import QuasiNewton, BFGS
def test_emt_h3o2m(initial, final, testdir):
    images = [initial.copy()]
    for i in range(3):
        images.append(initial.copy())
    images.append(final.copy())
    neb = NEB(images, climb=True)
    constraint = FixAtoms(indices=[1, 3])
    for image in images:
        image.calc = EMT()
        image.set_constraint(constraint)
    for image in images:
        print(image.get_distance(1, 2), image.get_potential_energy())
    dyn1 = QuasiNewton(images[0])
    dyn1.run(fmax=0.01)
    dyn2 = QuasiNewton(images[-1])
    dyn2.run(fmax=0.01)
    neb.interpolate()
    for image in images:
        print(image.get_distance(1, 2), image.get_potential_energy())
    with BFGS(neb, trajectory='emt_h3o2m.traj') as dyn:
        dyn.run(fmax=0.05)
    for image in images:
        print(image.get_distance(1, 2), image.get_potential_energy())