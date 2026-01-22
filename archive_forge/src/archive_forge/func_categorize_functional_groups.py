from __future__ import annotations
import copy
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.core.structure import Molecule
from pymatgen.io.babel import BabelMolAdaptor
def categorize_functional_groups(self, groups):
    """
        Determine classes of functional groups present in a set.

        Args:
            groups: Set of functional groups.

        Returns:
            dict containing representations of the groups, the indices of
            where the group occurs in the MoleculeGraph, and how many of each
            type of group there is.
        """
    categories = {}
    em = iso.numerical_edge_match('weight', 1)
    nm = iso.categorical_node_match('specie', 'C')
    for group in groups:
        atoms = [self.molecule[a] for a in group]
        species = [a.specie for a in atoms]
        coords = [a.coords for a in atoms]
        adaptor = BabelMolAdaptor(Molecule(species, coords))
        smiles = adaptor.pybel_mol.write('can').strip()
        if smiles in categories:
            this_subgraph = self.molgraph.graph.subgraph(list(group)).to_undirected()
            for other in categories[smiles]['groups']:
                other_subgraph = self.molgraph.graph.subgraph(list(other)).to_undirected()
                if not nx.is_isomorphic(this_subgraph, other_subgraph, edge_match=em, node_match=nm):
                    break
                if group not in categories[smiles]['groups']:
                    categories[smiles]['groups'].append(group)
                    categories[smiles]['count'] += 1
        else:
            categories[smiles] = {'groups': [group], 'count': 1}
    return categories