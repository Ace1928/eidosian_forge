from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import numpy as np
def get_symmetry_layerdataset(cell: Cell, aperiodic_dir=2, symprec=1e-05):
    """TODO: Add comments."""
    _set_no_error()
    lattice, positions, numbers, _ = _expand_cell(cell)
    spg_ds = _spglib.layerdataset(lattice, positions, numbers, aperiodic_dir, symprec)
    if spg_ds is None:
        _set_error_message()
        return None
    dataset = _build_dataset_dict(spg_ds)
    _set_error_message()
    return dataset