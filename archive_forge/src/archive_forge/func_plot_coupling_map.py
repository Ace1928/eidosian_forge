import math
from typing import List
import numpy as np
import rustworkx as rx
from rustworkx.visualization import graphviz_draw
from qiskit.exceptions import QiskitError
from qiskit.utils import optionals as _optionals
from qiskit.providers.exceptions import BackendPropertyError
from qiskit.transpiler.coupling import CouplingMap
from .exceptions import VisualizationError
@_optionals.HAS_MATPLOTLIB.require_in_call
@_optionals.HAS_GRAPHVIZ.require_in_call
def plot_coupling_map(num_qubits: int, qubit_coordinates: List[List[int]], coupling_map: List[List[int]], figsize=None, plot_directed=False, label_qubits=True, qubit_size=None, line_width=4, font_size=None, qubit_color=None, qubit_labels=None, line_color=None, font_color='white', ax=None, filename=None):
    """Plots an arbitrary coupling map of qubits (embedded in a plane).

    Args:
        num_qubits (int): The number of qubits defined and plotted.
        qubit_coordinates (List[List[int]]): A list of two-element lists, with entries of each nested
            list being the planar coordinates in a 0-based square grid where each qubit is located.
        coupling_map (List[List[int]]): A list of two-element lists, with entries of each nested
            list being the qubit numbers of the bonds to be plotted.
        figsize (tuple): Output figure size (wxh) in inches.
        plot_directed (bool): Plot directed coupling map.
        label_qubits (bool): Label the qubits.
        qubit_size (float): Size of qubit marker.
        line_width (float): Width of lines.
        font_size (int): Font size of qubit labels.
        qubit_color (list): A list of colors for the qubits
        qubit_labels (list): A list of qubit labels
        line_color (list): A list of colors for each line from coupling_map.
        font_color (str): The font color for the qubit labels.
        ax (Axes): A Matplotlib axes instance.
        filename (str): file path to save image to.

    Returns:
        Figure: A Matplotlib figure instance.

    Raises:
        MissingOptionalLibraryError: If matplotlib or graphviz is not installed.
        QiskitError: If length of qubit labels does not match number of qubits.

    Example:

        .. plot::
           :include-source:

            from qiskit.visualization import plot_coupling_map

            num_qubits = 8
            qubit_coordinates = [[0, 1], [1, 1], [1, 0], [1, 2], [2, 0], [2, 2], [2, 1], [3, 1]]
            coupling_map = [[0, 1], [1, 2], [2, 3], [3, 5], [4, 5], [5, 6], [2, 4], [6, 7]]
            plot_coupling_map(num_qubits, qubit_coordinates, coupling_map)
    """
    import matplotlib.pyplot as plt
    from .utils import matplotlib_close_if_inline
    input_axes = False
    if ax:
        input_axes = True
    if qubit_size is None:
        qubit_size = 30
    if qubit_labels is None:
        qubit_labels = list(range(num_qubits))
    elif len(qubit_labels) != num_qubits:
        raise QiskitError('Length of qubit labels does not equal number of qubits.')
    if not label_qubits:
        qubit_labels = [''] * num_qubits
    if qubit_color is None:
        qubit_color = ['#648fff'] * num_qubits
    if line_color is None:
        line_color = ['#648fff'] * len(coupling_map)
    if num_qubits == 1:
        graph = rx.PyDiGraph()
        graph.add_node(0)
    else:
        graph = CouplingMap(coupling_map).graph
    if not plot_directed:
        graph = graph.to_undirected(multigraph=False)
    for node in graph.node_indices():
        graph[node] = node
    for edge_index in graph.edge_indices():
        graph.update_edge_by_index(edge_index, edge_index)
    px = 1.15 / plt.rcParams['figure.dpi']
    if qubit_coordinates:
        qubit_coordinates = [coordinates[::-1] for coordinates in qubit_coordinates]
    if font_size is None:
        max_characters = max(1, max((len(str(x)) for x in qubit_labels)))
        font_size = max(int(20 / max_characters), 1)

    def color_node(node):
        if qubit_coordinates:
            out_dict = {'label': str(qubit_labels[node]), 'color': f'"{qubit_color[node]}"', 'fillcolor': f'"{qubit_color[node]}"', 'style': 'filled', 'shape': 'circle', 'pos': f'"{qubit_coordinates[node][0]},{qubit_coordinates[node][1]}"', 'pin': 'True'}
        else:
            out_dict = {'label': str(qubit_labels[node]), 'color': f'"{qubit_color[node]}"', 'fillcolor': f'"{qubit_color[node]}"', 'style': 'filled', 'shape': 'circle'}
        out_dict['fontcolor'] = f'"{font_color}"'
        out_dict['fontsize'] = str(font_size)
        out_dict['height'] = str(qubit_size * px)
        out_dict['fixedsize'] = 'True'
        out_dict['fontname'] = '"DejaVu Sans"'
        return out_dict

    def color_edge(edge):
        out_dict = {'color': f'"{line_color[edge]}"', 'fillcolor': f'"{line_color[edge]}"', 'penwidth': str(line_width)}
        return out_dict
    plot = graphviz_draw(graph, method='neato', node_attr_fn=color_node, edge_attr_fn=color_edge, filename=filename)
    if filename:
        return None
    if not input_axes:
        if figsize is None:
            width, height = plot.size
            figsize = (width * px, height * px)
        fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    ax.imshow(plot)
    if not input_axes:
        matplotlib_close_if_inline(fig)
        return fig