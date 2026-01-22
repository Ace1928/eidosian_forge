from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pymatgen
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN, metal_edge_extender
from pymatgen.core import Element, Molecule
def create_openff_mol(smile: str, geometry: pymatgen.core.Molecule | str | Path | None=None, charge_scaling: float=1, partial_charges: list[float] | None=None, backup_charge_method: str='am1bcc') -> tk.Molecule:
    """
    Create an OpenFF Molecule from a SMILES string and optional geometry.

    Constructs an OpenFF Molecule from the provided SMILES
    string, adds conformers based on the provided geometry (if
    any), assigns partial charges using the specified method
    or provided partial charges, and applies charge scaling.

    Args:
        smile (str): The SMILES string of the molecule.
        geometry (Union[pymatgen.core.Molecule, str, Path, None], optional): The
            geometry to use for adding conformers. Can be a Pymatgen Molecule,
            file path, or None.
        charge_scaling (float, optional): The scaling factor for partial charges.
            Default is 1.
        partial_charges (Union[List[float], None], optional): A list of partial
            charges to assign, or None to use the charge method.
        backup_charge_method (str, optional): The backup charge method to use if
            partial charges are not provided. Default is "am1bcc".

    Returns:
        tk.Molecule: The created OpenFF Molecule.
    """
    if isinstance(geometry, (str, Path)):
        geometry = pymatgen.core.Molecule.from_file(str(geometry))
    if partial_charges is not None:
        if geometry is None:
            raise ValueError('geometries must be set if partial_charges is set')
        if len(partial_charges) != len(geometry):
            raise ValueError('partial charges must have same length & order as geometry')
    openff_mol = tk.Molecule.from_smiles(smile, allow_undefined_stereo=True)
    openff_mol, atom_map = add_conformer(openff_mol, geometry)
    openff_mol = assign_partial_charges(openff_mol, atom_map, backup_charge_method, partial_charges)
    openff_mol.partial_charges *= charge_scaling
    return openff_mol