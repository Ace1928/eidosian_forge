import numpy as np
from ase.calculators.calculator import Calculator
from ase.data import atomic_numbers
from ase.utils import IOContext
from ase.geometry import get_distances
from ase.cell import Cell
def get_region_from_masks(self, atoms=None, print_mapping=False):
    """
        creates region array from the masks of the calculators. The tags in
        the array are:
        QM - qm atoms
        buffer - buffer atoms
        MM - atoms treated with mm calculator
        """
    if atoms is None:
        if self.atoms is None:
            raise ValueError('Calculator has no atoms')
        else:
            atoms = self.atoms
    region = np.full_like(atoms, 'MM')
    region[self.qm_selection_mask] = np.full_like(region[self.qm_selection_mask], 'QM')
    buffer_only_mask = self.qm_buffer_mask & ~self.qm_selection_mask
    region[buffer_only_mask] = np.full_like(region[buffer_only_mask], 'buffer')
    if print_mapping:
        print(f'Mapping of {len(region):5d} atoms in total:')
        for region_id in np.unique(region):
            n_at = np.count_nonzero(region == region_id)
            print(f'{n_at:16d} {region_id}')
        qm_atoms = atoms[self.qm_selection_mask]
        symbol_counts = qm_atoms.symbols.formula.count()
        print('QM atoms types:')
        for symbol, count in symbol_counts.items():
            print(f'{count:16d} {symbol}')
    return region