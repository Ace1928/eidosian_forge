from __future__ import annotations
import copy
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.core.structure import Molecule
from pymatgen.io.babel import BabelMolAdaptor
def get_special_carbon(self, elements=None):
    """
        Identify Carbon atoms in the MoleculeGraph that fit the characteristics
        defined Ertl (2017), returning a list of their node indices.

        The conditions for marking carbon atoms are (quoted from Ertl):
            "- atoms connected by non-aromatic double or triple bond to any
            heteroatom
            - atoms in nonaromatic carbon-carbon double or triple bonds
            - acetal carbons, i.e. sp3 carbons connected to two or more oxygens,
            nitrogens or sulfurs; these O, N or S atoms must have only single bonds
            - all atoms in oxirane, aziridine and thiirane rings"

        Args:
            elements: List of elements that will qualify a carbon as special
                (if only certain functional groups are of interest).
                Default None.

        Returns:
            set of ints representing node indices
        """
    specials = set()
    carbons = [n for n in self.molgraph.graph.nodes if str(self.species[n]) == 'C']
    for node in carbons:
        neighbors = self.molgraph.graph[node]
        for neighbor, attributes in neighbors.items():
            if elements is not None:
                if str(self.species[neighbor]) in elements and int(attributes[0]['weight']) in [2, 3]:
                    specials.add(node)
            elif str(self.species[neighbor]) not in ['C', 'H'] and int(attributes[0]['weight']) in [2, 3]:
                specials.add(node)
    for node in carbons:
        neighbors = self.molgraph.graph[node]
        for neighbor, attributes in neighbors.items():
            if str(self.species[neighbor]) == 'C' and int(attributes[0]['weight']) in [2, 3]:
                specials.add(node)
                specials.add(neighbor)
    for node in carbons:
        neighbors = self.molgraph.graph[node]
        neighbor_spec = [str(self.species[n]) for n in neighbors]
        ons = len([n for n in neighbor_spec if n in ['O', 'N', 'S']])
        if len(neighbors) == 4 and ons >= 2:
            specials.add(node)
    rings = self.molgraph.find_rings()
    rings_indices = [set(sum(ring, ())) for ring in rings]
    for ring in rings_indices:
        ring_spec = sorted((str(self.species[node]) for node in ring))
        if len(ring) == 3 and ring_spec in [['C', 'C', 'O'], ['C', 'C', 'N'], ['C', 'C', 'S']]:
            for node in ring:
                if node in carbons:
                    specials.add(node)
    return specials