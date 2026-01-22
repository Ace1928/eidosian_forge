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
def print_links(self) -> None:
    """Print all links in the graph."""
    nodes = self.environment_subgraph().nodes()
    print('Links in graph :')
    for node in nodes:
        print(node.isite, ' is connected with : ')
        for n1, n2, data in self.environment_subgraph().edges(node, data=True):
            if n1.isite == data['start']:
                print(f'  - {n2.isite} by {len(data['ligands'])} ligands ({data['delta'][0]} {data['delta'][1]} {data['delta'][2]})')
            else:
                print(f'  - {n2.isite} by {len(data['ligands'])} ligands ({-data['delta'][0]} {-data['delta'][1]} {-data['delta'][2]})')