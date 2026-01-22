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
def search_fragment_entries(self, frag) -> list:
    """
        Search all fragment entries for those isomorphic to the given fragment.
        We distinguish between entries where both initial and final MoleculeGraphs are isomorphic to the
        given fragment (entries) vs those where only the initial MoleculeGraph is isomorphic to the given
        fragment (initial_entries) vs those where only the final MoleculeGraph is isomorphic (final_entries).

        Args:
            frag: Fragment
        """
    entries = []
    initial_entries = []
    final_entries = []
    for entry in self.filtered_entries:
        if frag.isomorphic_to(entry['initial_molgraph']) and frag.isomorphic_to(entry['final_molgraph']):
            entries += [entry]
        elif frag.isomorphic_to(entry['initial_molgraph']):
            initial_entries += [entry]
        elif frag.isomorphic_to(entry['final_molgraph']):
            final_entries += [entry]
    return [entries, initial_entries, final_entries]