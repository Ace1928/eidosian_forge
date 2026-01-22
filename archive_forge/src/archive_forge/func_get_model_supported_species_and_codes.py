import numpy as np
from ase.calculators.calculator import Calculator
from ase.calculators.calculator import compare_atoms
from . import kimpy_wrappers
from . import neighborlist
def get_model_supported_species_and_codes(self):
    return self.kimmodeldata.get_model_supported_species_and_codes