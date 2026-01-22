from enum import Enum
import itertools
import logging
import time
from rustworkx import vf2_mapping
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.layout import vf2_utils
def mapping_to_layout(layout_mapping):
    return Layout({reverse_im_graph_node_map[k]: v for k, v in layout_mapping.items()})