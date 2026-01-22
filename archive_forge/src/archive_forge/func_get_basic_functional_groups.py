from __future__ import annotations
import copy
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.core.structure import Molecule
from pymatgen.io.babel import BabelMolAdaptor
def get_basic_functional_groups(self, func_groups=None):
    """
        Identify functional groups that cannot be identified by the Ertl method
        of get_special_carbon and get_heteroatoms, such as benzene rings, methyl
        groups, and ethyl groups.

        TODO: Think of other functional groups that are important enough to be
        added (ex: do we need ethyl, butyl, propyl?)

        Args:
            func_groups: List of strs representing the functional groups of
                interest. Default to None, meaning that all of the functional groups
                defined in this function will be sought.

        Returns:
            list of sets of ints, representing groups of connected atoms
        """
    strat = OpenBabelNN()
    hydrogens = {n for n in self.molgraph.graph.nodes if str(self.species[n]) == 'H'}
    carbons = [n for n in self.molgraph.graph.nodes if str(self.species[n]) == 'C']
    if func_groups is None:
        func_groups = ['methyl', 'phenyl']
    results = []
    if 'methyl' in func_groups:
        for node in carbons:
            neighbors = strat.get_nn_info(self.molecule, node)
            hs = {n['site_index'] for n in neighbors if n['site_index'] in hydrogens}
            if len(hs) >= 3:
                hs.add(node)
                results.append(hs)
    if 'phenyl' in func_groups:
        rings_indices = [set(sum(ring, ())) for ring in self.molgraph.find_rings()]
        possible_phenyl = [r for r in rings_indices if len(r) == 6]
        for ring in possible_phenyl:
            num_deviants = 0
            for node in ring:
                neighbors = strat.get_nn_info(self.molecule, node)
                neighbor_spec = sorted((str(self.species[n['site_index']]) for n in neighbors))
                if neighbor_spec != ['C', 'C', 'H']:
                    num_deviants += 1
            if num_deviants <= 1:
                for node in ring:
                    ring_group = copy.deepcopy(ring)
                    neighbors = self.molgraph.graph[node]
                    for neighbor in neighbors:
                        if neighbor in hydrogens:
                            ring_group.add(neighbor)
                results.append(ring_group)
    return results