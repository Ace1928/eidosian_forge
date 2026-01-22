from __future__ import annotations
import collections
import logging
from typing import TYPE_CHECKING
import networkx as nx
import numpy as np
from monty.json import MSONable, jsanitize
from pymatgen.analysis.chemenv.connectivity.connected_components import ConnectedComponent
from pymatgen.analysis.chemenv.connectivity.environment_nodes import get_environment_node
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments
def setup_environment_subgraph(self, environments_symbols, only_atoms=None):
    """
        Set up the graph for predefined environments and optionally atoms.

        Args:
            environments_symbols: Symbols of the environments for the environment subgraph.
            only_atoms: Atoms to be considered.
        """
    logging.info(f'Setup of environment subgraph for environments {', '.join(environments_symbols)}')
    if not isinstance(environments_symbols, collections.abc.Iterable):
        environments_symbols = [environments_symbols]
    environments_symbols = sorted(environments_symbols)
    envs_string = '-'.join(environments_symbols)
    if only_atoms is not None:
        envs_string += '#' + '-'.join(sorted(only_atoms))
    if envs_string in self.environment_subgraphs:
        self._environment_subgraph = self.environment_subgraphs[envs_string]
        return
    self._environment_subgraph = nx.MultiGraph()
    for isite, ce_this_site_all in enumerate(self.light_structure_environments.coordination_environments):
        if ce_this_site_all is None:
            continue
        if len(ce_this_site_all) == 0:
            continue
        ce_this_site = ce_this_site_all[0]['ce_symbol']
        if ce_this_site in environments_symbols:
            if only_atoms is None:
                env_node = get_environment_node(self.light_structure_environments.structure[isite], isite, ce_this_site)
                self._environment_subgraph.add_node(env_node)
            elif self.light_structure_environments.structure.is_ordered:
                if self.light_structure_environments.structure[isite].specie.symbol in only_atoms:
                    env_node = get_environment_node(self.light_structure_environments.structure[isite], isite, ce_this_site)
                    self._environment_subgraph.add_node(env_node)
            else:
                this_site_elements = [sp.symbol for sp in self.light_structure_environments.structure[isite].species_and_occu]
                for elem_symbol in this_site_elements:
                    if elem_symbol in only_atoms:
                        env_node = get_environment_node(self.light_structure_environments.structure[isite], isite, ce_this_site)
                        self._environment_subgraph.add_node(env_node)
                        break
    nodes = list(self._environment_subgraph.nodes())
    for inode1, node1 in enumerate(nodes):
        isite1 = node1.isite
        links_node1 = self._graph.edges(isite1, data=True)
        for node2 in nodes[inode1:]:
            isite2 = node2.isite
            links_node2 = self._graph.edges(isite2, data=True)
            connections_site1_site2 = {}
            for _, ilig_site1, d1 in links_node1:
                for _, ilig_site2, d2 in links_node2:
                    if ilig_site1 == ilig_site2:
                        delta_image = get_delta_image(isite1, isite2, d1, d2)
                        if isite1 == isite2 and np.all(delta_image == 0):
                            continue
                        tuple_delta_image = tuple(delta_image)
                        if tuple_delta_image in connections_site1_site2:
                            connections_site1_site2[tuple_delta_image].append((ilig_site1, d1, d2))
                        else:
                            connections_site1_site2[tuple_delta_image] = [(ilig_site1, d1, d2)]
            if isite1 == isite2:
                remove_deltas = []
                alldeltas = list(connections_site1_site2)
                alldeltas2 = list(connections_site1_site2)
                if (0, 0, 0) in alldeltas:
                    alldeltas.remove((0, 0, 0))
                    alldeltas2.remove((0, 0, 0))
                for current_delta in alldeltas:
                    opp_current_delta = tuple((-dd for dd in current_delta))
                    if opp_current_delta in alldeltas2:
                        remove_deltas.append(current_delta)
                        alldeltas2.remove(current_delta)
                        alldeltas2.remove(opp_current_delta)
                for remove_delta in remove_deltas:
                    connections_site1_site2.pop(remove_delta)
            for conn, ligands in list(connections_site1_site2.items()):
                self._environment_subgraph.add_edge(node1, node2, start=node1.isite, end=node2.isite, delta=conn, ligands=ligands)
    self.environment_subgraphs[envs_string] = self._environment_subgraph