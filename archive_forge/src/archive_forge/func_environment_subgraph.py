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
def environment_subgraph(self, environments_symbols=None, only_atoms=None):
    """
        Args:
            environments_symbols ():
            only_atoms ():

        Returns:
            nx.MultiGraph: The subgraph of the structure connectivity graph
        """
    if environments_symbols is not None:
        self.setup_environment_subgraph(environments_symbols=environments_symbols, only_atoms=only_atoms)
    try:
        return self._environment_subgraph
    except AttributeError:
        all_envs = self.light_structure_environments.environments_identified()
        self.setup_environment_subgraph(environments_symbols=all_envs, only_atoms=only_atoms)
    return self._environment_subgraph