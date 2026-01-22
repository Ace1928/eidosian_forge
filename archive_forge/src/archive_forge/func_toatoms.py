from random import randint
from typing import Dict, Tuple, Any
import numpy as np
from ase import Atoms
from ase.constraints import dict2constraint
from ase.calculators.calculator import (get_calculator_class, all_properties,
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import chemical_symbols, atomic_masses
from ase.formula import Formula
from ase.geometry import cell_to_cellpar
from ase.io.jsonio import decode
def toatoms(self, attach_calculator=False, add_additional_information=False):
    """Create Atoms object."""
    atoms = Atoms(self.numbers, self.positions, cell=self.cell, pbc=self.pbc, magmoms=self.get('initial_magmoms'), charges=self.get('initial_charges'), tags=self.get('tags'), masses=self.get('masses'), momenta=self.get('momenta'), constraint=self.constraints)
    if attach_calculator:
        params = self.get('calculator_parameters', {})
        atoms.calc = get_calculator_class(self.calculator)(**params)
    else:
        results = {}
        for prop in all_properties:
            if prop in self:
                results[prop] = self[prop]
        if results:
            atoms.calc = SinglePointCalculator(atoms, **results)
            atoms.calc.name = self.get('calculator', 'unknown')
    if add_additional_information:
        atoms.info = {}
        atoms.info['unique_id'] = self.unique_id
        if self._keys:
            atoms.info['key_value_pairs'] = self.key_value_pairs
        data = self.get('data')
        if data:
            atoms.info['data'] = data
    return atoms