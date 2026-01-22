from __future__ import annotations
import copy
import logging
from typing import TYPE_CHECKING
from monty.json import MSONable
from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError
from pymatgen.analysis.local_env import OpenBabelNN, metal_edge_extender
from pymatgen.io.babel import BabelMolAdaptor
def open_ring(mol_graph: MoleculeGraph, bond: list, opt_steps: int) -> MoleculeGraph:
    """
    Function to actually open a ring using OpenBabel's local opt. Given a molecule
    graph and a bond, convert the molecule graph into an OpenBabel molecule, remove
    the given bond, perform the local opt with the number of steps determined by
    self.steps, and then convert the resulting structure back into a molecule graph
    to be returned.
    """
    ob_mol = BabelMolAdaptor.from_molecule_graph(mol_graph)
    ob_mol.remove_bond(bond[0][0] + 1, bond[0][1] + 1)
    ob_mol.localopt(steps=opt_steps, forcefield='uff')
    return MoleculeGraph.from_local_env_strategy(ob_mol.pymatgen_mol, OpenBabelNN())