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