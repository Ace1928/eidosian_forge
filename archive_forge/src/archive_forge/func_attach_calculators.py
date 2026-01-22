from ase.autoneb import AutoNEB
from ase.build import fcc211, add_adsorbate
from ase.constraints import FixAtoms
from ase.neb import NEBTools
from ase.optimize import QuasiNewton
def attach_calculators(images):
    for i in range(len(images)):
        images[i].calc = EMT()