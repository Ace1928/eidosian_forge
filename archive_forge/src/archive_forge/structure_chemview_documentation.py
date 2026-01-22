from __future__ import annotations
import numpy as np
from monty.dev import requires
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    A function to visualize pymatgen Structure objects in jupyter notebook using chemview package.

    Args:
        structure: pymatgen Structure
        bonds: (bool) visualize bonds. Bonds are found by comparing distances
            to added covalent radii of pairs. Defaults to True.
        conventional: (bool) use conventional cell. Defaults to False.
        transform: (list) can be used to make supercells with pymatgen.Structure.make_supercell method
        show_box: (bool) unit cell is shown. Defaults to True.
        bond_tol: (float) used if bonds=True. Sets the extra distance tolerance when finding bonds.
        stick_radius: (float) radius of bonds.

    Returns:
        A chemview.MolecularViewer object
    