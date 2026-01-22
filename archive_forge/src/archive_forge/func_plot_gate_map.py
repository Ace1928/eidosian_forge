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
def plot_gate_map(backend, figsize=None, plot_directed=False, label_qubits=True, qubit_size=None, line_width=4, font_size=None, qubit_color=None, qubit_labels=None, line_color=None, font_color='white', ax=None, filename=None, qubit_coordinates=None):
    """Plots the gate map of a device.

    Args:
        backend (Backend): The backend instance that will be used to plot the device
            gate map.
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
        qubit_coordinates (Sequence): An optional sequence input (list or array being the
            most common) of 2d coordinates for each qubit. The length of the
            sequence much match the number of qubits on the backend. The sequence
            should be the planar coordinates in a 0-based square grid where each
            qubit is located.

    Returns:
        Figure: A Matplotlib figure instance.

    Raises:
        QiskitError: if tried to pass a simulator, or if the backend is None,
            but one of num_qubits, mpl_data, or cmap is None.
        MissingOptionalLibraryError: if matplotlib not installed.

    Example:

        .. plot::
           :include-source:

           from qiskit.providers.fake_provider import GenericBackendV2
           from qiskit.visualization import plot_gate_map

           backend = GenericBackendV2(num_qubits=5)

           plot_gate_map(backend)
    """
    qubit_coordinates_map = {}
    qubit_coordinates_map[5] = [[1, 0], [0, 1], [1, 1], [1, 2], [2, 1]]
    qubit_coordinates_map[7] = [[0, 0], [0, 1], [0, 2], [1, 1], [2, 0], [2, 1], [2, 2]]
    qubit_coordinates_map[20] = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [3, 0], [3, 1], [3, 2], [3, 3], [3, 4]]
    qubit_coordinates_map[15] = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [1, 7], [1, 6], [1, 5], [1, 4], [1, 3], [1, 2], [1, 1], [1, 0]]
    qubit_coordinates_map[16] = [[1, 0], [1, 1], [2, 1], [3, 1], [1, 2], [3, 2], [0, 3], [1, 3], [3, 3], [4, 3], [1, 4], [3, 4], [1, 5], [2, 5], [3, 5], [1, 6]]
    qubit_coordinates_map[27] = [[1, 0], [1, 1], [2, 1], [3, 1], [1, 2], [3, 2], [0, 3], [1, 3], [3, 3], [4, 3], [1, 4], [3, 4], [1, 5], [2, 5], [3, 5], [1, 6], [3, 6], [0, 7], [1, 7], [3, 7], [4, 7], [1, 8], [3, 8], [1, 9], [2, 9], [3, 9], [3, 10]]
    qubit_coordinates_map[28] = [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [1, 2], [1, 6], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [3, 0], [3, 4], [3, 8], [4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8]]
    qubit_coordinates_map[53] = [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [1, 2], [1, 6], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [3, 0], [3, 4], [3, 8], [4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8], [5, 2], [5, 6], [6, 0], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [6, 6], [6, 7], [6, 8], [7, 0], [7, 4], [7, 8], [8, 0], [8, 1], [8, 2], [8, 3], [8, 4], [8, 5], [8, 6], [8, 7], [8, 8], [9, 2], [9, 6]]
    qubit_coordinates_map[65] = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [1, 0], [1, 4], [1, 8], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9], [2, 10], [3, 2], [3, 6], [3, 10], [4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8], [4, 9], [4, 10], [5, 0], [5, 4], [5, 8], [6, 0], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [6, 6], [6, 7], [6, 8], [6, 9], [6, 10], [7, 2], [7, 6], [7, 10], [8, 1], [8, 2], [8, 3], [8, 4], [8, 5], [8, 6], [8, 7], [8, 8], [8, 9], [8, 10]]
    qubit_coordinates_map[127] = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [1, 0], [1, 4], [1, 8], [1, 12], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9], [2, 10], [2, 11], [2, 12], [2, 13], [2, 14], [3, 2], [3, 6], [3, 10], [3, 14], [4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8], [4, 9], [4, 10], [4, 11], [4, 12], [4, 13], [4, 14], [5, 0], [5, 4], [5, 8], [5, 12], [6, 0], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [6, 6], [6, 7], [6, 8], [6, 9], [6, 10], [6, 11], [6, 12], [6, 13], [6, 14], [7, 2], [7, 6], [7, 10], [7, 14], [8, 0], [8, 1], [8, 2], [8, 3], [8, 4], [8, 5], [8, 6], [8, 7], [8, 8], [8, 9], [8, 10], [8, 11], [8, 12], [8, 13], [8, 14], [9, 0], [9, 4], [9, 8], [9, 12], [10, 0], [10, 1], [10, 2], [10, 3], [10, 4], [10, 5], [10, 6], [10, 7], [10, 8], [10, 9], [10, 10], [10, 11], [10, 12], [10, 13], [10, 14], [11, 2], [11, 6], [11, 10], [11, 14], [12, 1], [12, 2], [12, 3], [12, 4], [12, 5], [12, 6], [12, 7], [12, 8], [12, 9], [12, 10], [12, 11], [12, 12], [12, 13], [12, 14]]
    qubit_coordinates_map[433] = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [0, 14], [0, 15], [0, 16], [0, 17], [0, 18], [0, 19], [0, 20], [0, 21], [0, 22], [0, 23], [0, 24], [0, 25], [1, 0], [1, 4], [1, 8], [1, 12], [1, 16], [1, 20], [1, 24], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9], [2, 10], [2, 11], [2, 12], [2, 13], [2, 14], [2, 15], [2, 16], [2, 17], [2, 18], [2, 19], [2, 20], [2, 21], [2, 22], [2, 23], [2, 24], [2, 25], [2, 26], [3, 2], [3, 6], [3, 10], [3, 14], [3, 18], [3, 22], [3, 26], [4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8], [4, 9], [4, 10], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 16], [4, 17], [4, 18], [4, 19], [4, 20], [4, 21], [4, 22], [4, 23], [4, 24], [4, 25], [4, 26], [5, 0], [5, 4], [5, 8], [5, 12], [5, 16], [5, 20], [5, 24], [6, 0], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [6, 6], [6, 7], [6, 8], [6, 9], [6, 10], [6, 11], [6, 12], [6, 13], [6, 14], [6, 15], [6, 16], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 22], [6, 23], [6, 24], [6, 25], [6, 26], [7, 2], [7, 6], [7, 10], [7, 14], [7, 18], [7, 22], [7, 26], [8, 0], [8, 1], [8, 2], [8, 3], [8, 4], [8, 5], [8, 6], [8, 7], [8, 8], [8, 9], [8, 10], [8, 11], [8, 12], [8, 13], [8, 14], [8, 15], [8, 16], [8, 17], [8, 18], [8, 19], [8, 20], [8, 21], [8, 22], [8, 23], [8, 24], [8, 25], [8, 26], [9, 0], [9, 4], [9, 8], [9, 12], [9, 16], [9, 20], [9, 24], [10, 0], [10, 1], [10, 2], [10, 3], [10, 4], [10, 5], [10, 6], [10, 7], [10, 8], [10, 9], [10, 10], [10, 11], [10, 12], [10, 13], [10, 14], [10, 15], [10, 16], [10, 17], [10, 18], [10, 19], [10, 20], [10, 21], [10, 22], [10, 23], [10, 24], [10, 25], [10, 26], [11, 2], [11, 6], [11, 10], [11, 14], [11, 18], [11, 22], [11, 26], [12, 0], [12, 1], [12, 2], [12, 3], [12, 4], [12, 5], [12, 6], [12, 7], [12, 8], [12, 9], [12, 10], [12, 11], [12, 12], [12, 13], [12, 14], [12, 15], [12, 16], [12, 17], [12, 18], [12, 19], [12, 20], [12, 21], [12, 22], [12, 23], [12, 24], [12, 25], [12, 26], [13, 0], [13, 4], [13, 8], [13, 12], [13, 16], [13, 20], [13, 24], [14, 0], [14, 1], [14, 2], [14, 3], [14, 4], [14, 5], [14, 6], [14, 7], [14, 8], [14, 9], [14, 10], [14, 11], [14, 12], [14, 13], [14, 14], [14, 15], [14, 16], [14, 17], [14, 18], [14, 19], [14, 20], [14, 21], [14, 22], [14, 23], [14, 24], [14, 25], [14, 26], [15, 2], [15, 6], [15, 10], [15, 14], [15, 18], [15, 22], [15, 26], [16, 0], [16, 1], [16, 2], [16, 3], [16, 4], [16, 5], [16, 6], [16, 7], [16, 8], [16, 9], [16, 10], [16, 11], [16, 12], [16, 13], [16, 14], [16, 15], [16, 16], [16, 17], [16, 18], [16, 19], [16, 20], [16, 21], [16, 22], [16, 23], [16, 24], [16, 25], [16, 26], [17, 0], [17, 4], [17, 8], [17, 12], [17, 16], [17, 20], [17, 24], [18, 0], [18, 1], [18, 2], [18, 3], [18, 4], [18, 5], [18, 6], [18, 7], [18, 8], [18, 9], [18, 10], [18, 11], [18, 12], [18, 13], [18, 14], [18, 15], [18, 16], [18, 17], [18, 18], [18, 19], [18, 20], [18, 21], [18, 22], [18, 23], [18, 24], [18, 25], [18, 26], [19, 2], [19, 6], [19, 10], [19, 14], [19, 18], [19, 22], [19, 26], [20, 0], [20, 1], [20, 2], [20, 3], [20, 4], [20, 5], [20, 6], [20, 7], [20, 8], [20, 9], [20, 10], [20, 11], [20, 12], [20, 13], [20, 14], [20, 15], [20, 16], [20, 17], [20, 18], [20, 19], [20, 20], [20, 21], [20, 22], [20, 23], [20, 24], [20, 25], [20, 26], [21, 0], [21, 4], [21, 8], [21, 12], [21, 16], [21, 20], [21, 24], [22, 0], [22, 1], [22, 2], [22, 3], [22, 4], [22, 5], [22, 6], [22, 7], [22, 8], [22, 9], [22, 10], [22, 11], [22, 12], [22, 13], [22, 14], [22, 15], [22, 16], [22, 17], [22, 18], [22, 19], [22, 20], [22, 21], [22, 22], [22, 23], [22, 24], [22, 25], [22, 26], [23, 2], [23, 6], [23, 10], [23, 14], [23, 18], [23, 22], [23, 26], [24, 1], [24, 2], [24, 3], [24, 4], [24, 5], [24, 6], [24, 7], [24, 8], [24, 9], [24, 10], [24, 11], [24, 12], [24, 13], [24, 14], [24, 15], [24, 16], [24, 17], [24, 18], [24, 19], [24, 20], [24, 21], [24, 22], [24, 23], [24, 24], [24, 25], [24, 26]]
    backend_version = _get_backend_interface_version(backend)
    if backend_version <= 1:
        if backend.configuration().simulator:
            raise QiskitError('Requires a device backend, not simulator.')
        config = backend.configuration()
        num_qubits = config.n_qubits
        coupling_map = CouplingMap(config.coupling_map)
        name = backend.name()
    else:
        num_qubits = backend.num_qubits
        coupling_map = backend.coupling_map
        name = backend.name
    if qubit_coordinates is None and ('ibm' in name or 'fake' in name):
        qubit_coordinates = qubit_coordinates_map.get(num_qubits, None)
    if qubit_coordinates:
        if len(qubit_coordinates) != num_qubits:
            raise QiskitError(f'The number of specified qubit coordinates {len(qubit_coordinates)} does not match the device number of qubits: {num_qubits}')
    return plot_coupling_map(num_qubits, qubit_coordinates, coupling_map.get_edges(), figsize, plot_directed, label_qubits, qubit_size, line_width, font_size, qubit_color, qubit_labels, line_color, font_color, ax, filename)