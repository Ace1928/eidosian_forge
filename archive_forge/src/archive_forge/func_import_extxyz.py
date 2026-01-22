import numpy as np
from ase.calculators.calculator import Calculator
from ase.data import atomic_numbers
from ase.utils import IOContext
from ase.geometry import get_distances
from ase.cell import Cell
@classmethod
def import_extxyz(cls, filename, qm_calc, mm_calc):
    """
        A static method to import the the mapping from an estxyz file saved by
        export_extxyz() function
        Parameters
        ----------
        filename: string
            filename with saved configuration

        qm_calc: Calculator object
            QM-calculator.
        mm_calc: Calculator object
            MM-calculator (should be scaled, see :class:`RescaledCalculator`)
            Can use `ForceConstantCalculator` based on QM force constants, if
            available.

        Returns
        -------
        New object of ForceQMMM calculator with qm_selection_mask and
        qm_buffer_mask set according to the region array in the saved file
        """
    from ase.io import read
    atoms = read(filename, format='extxyz')
    if 'region' in atoms.arrays:
        region = atoms.get_array('region')
    else:
        raise RuntimeError("Please provide extxyz file with 'region' array")
    dummy_qm_mask = np.full_like(atoms, False, dtype=bool)
    dummy_qm_mask[0] = True
    self = cls(atoms, dummy_qm_mask, qm_calc, mm_calc, buffer_width=1.0)
    self.set_masks_from_region(region)
    return self