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
def color_edge(edge):
    out_dict = {'color': f'"{line_color[edge]}"', 'fillcolor': f'"{line_color[edge]}"', 'penwidth': str(line_width)}
    return out_dict