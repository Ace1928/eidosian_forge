from __future__ import annotations
import logging
import warnings
import networkx as nx
from monty.json import MSONable
from pymatgen.analysis.fragmenter import open_ring
from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.core.structure import Molecule
from pymatgen.io.babel import BabelMolAdaptor
def build_new_entry(self, frags: list, bonds: list) -> list:
    """
        Build a new entry for bond dissociation that will be returned to the user.

        Args:
            frags (list): Fragments involved in the bond dissociation.
            bonds (list): Bonds broken in the dissociation process.

        Returns:
            list: Formatted bond dissociation entries.
        """
    specie = nx.get_node_attributes(self.mol_graph.graph, 'specie')
    if len(frags) == 2:
        new_entry = [self.molecule_entry['final_energy'] - (frags[0]['final_energy'] + frags[1]['final_energy']), bonds, specie[bonds[0][0]], specie[bonds[0][1]], frags[0]['smiles'], frags[0]['structure_change'], frags[0]['initial_molecule']['charge'], frags[0]['initial_molecule']['spin_multiplicity'], frags[0]['final_energy'], frags[1]['smiles'], frags[1]['structure_change'], frags[1]['initial_molecule']['charge'], frags[1]['initial_molecule']['spin_multiplicity'], frags[1]['final_energy']]
    else:
        new_entry = [self.molecule_entry['final_energy'] - frags[0]['final_energy'], bonds, specie[bonds[0][0]], specie[bonds[0][1]], frags[0]['smiles'], frags[0]['structure_change'], frags[0]['initial_molecule']['charge'], frags[0]['initial_molecule']['spin_multiplicity'], frags[0]['final_energy']]
    return new_entry