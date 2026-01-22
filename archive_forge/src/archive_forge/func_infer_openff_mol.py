from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pymatgen
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN, metal_edge_extender
from pymatgen.core import Element, Molecule
def infer_openff_mol(mol_geometry: pymatgen.core.Molecule) -> tk.Molecule:
    """Infer an OpenFF Molecule from a Pymatgen Molecule.

    Constructs a MoleculeGraph from the Pymatgen Molecule using the OpenBabelNN local
    environment strategy and extends metal edges. Converts the resulting MoleculeGraph
    to an OpenFF Molecule using mol_graph_to_openff_mol.

    Args:
        mol_geometry (pymatgen.core.Molecule): The Pymatgen Molecule to infer from.

    Returns:
        tk.Molecule: The inferred OpenFF Molecule.
    """
    mol_graph = MoleculeGraph.with_local_env_strategy(mol_geometry, OpenBabelNN())
    mol_graph = metal_edge_extender(mol_graph)
    return mol_graph_to_openff_mol(mol_graph)