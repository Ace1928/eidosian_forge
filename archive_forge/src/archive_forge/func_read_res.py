import glob
import re
from ase.atoms import Atoms
from ase.geometry import cellpar_to_cell, cell_to_cellpar
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
def read_res(filename, index=-1):
    """
    Read input in SHELX (.res) format

    Multiple frames are read if `filename` contains a wildcard character,
    e.g. `file_*.res`. `index` specifes which frames to retun: default is
    last frame only (index=-1).
    """
    images = []
    for fn in sorted(glob.glob(filename)):
        res = Res.from_file(fn)
        if res.energy:
            calc = SinglePointCalculator(res.atoms, energy=res.energy)
            res.atoms.calc = calc
        images.append(res.atoms)
    return images[index]