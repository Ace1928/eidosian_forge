from __future__ import annotations
import itertools
import logging
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch
from monty.json import MSONable, jsanitize
from networkx.algorithms.components import is_connected
from networkx.algorithms.traversal import bfs_tree
from pymatgen.analysis.chemenv.connectivity.environment_nodes import EnvironmentNode
from pymatgen.analysis.chemenv.utils.chemenv_errors import ChemenvError
from pymatgen.analysis.chemenv.utils.graph_utils import get_delta
from pymatgen.analysis.chemenv.utils.math_utils import get_linearly_independent_vectors
def show_graph(self, graph: nx.MultiGraph | None=None, save_file: str | None=None, drawing_type: str='internal') -> None:
    """
        Displays the graph using the specified drawing type.

        Args:
            graph (Graph, optional): The graph to display. If not provided, the current graph is used.
            save_file (str, optional): The file path to save the graph image to.
                If not provided, the graph is not saved.
            drawing_type (str): The type of drawing to use. Can be "internal" or "external".
        """
    shown_graph = self._connected_subgraph if graph is None else graph
    plt.figure()
    if drawing_type == 'internal':
        pos = nx.shell_layout(shown_graph)
        ax = plt.gca()
        draw_network(shown_graph, pos, ax, periodicity_vectors=self._periodicity_vectors)
        ax.autoscale()
        plt.axis('equal')
        plt.axis('off')
        if save_file is not None:
            plt.savefig(save_file)
    elif drawing_type == 'draw_graphviz':
        nx.nx_pydot.graphviz_layout(shown_graph)
    elif drawing_type == 'draw_random':
        nx.draw_random(shown_graph)