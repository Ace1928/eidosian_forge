from collections import defaultdict
import statistics
import random
import numpy as np
from rustworkx import PyDiGraph, PyGraph, connected_components
from qiskit.circuit import ControlFlowOp, ForLoopOp
from qiskit.converters import circuit_to_dag
from qiskit._accelerate import vf2_layout
from qiskit._accelerate.nlayout import NLayout
from qiskit._accelerate.error_map import ErrorMap
def build_edge_list(im_graph):
    """Generate an edge list for scoring."""
    return vf2_layout.EdgeList([((edge[0], edge[1]), sum(edge[2].values())) for edge in im_graph.edge_index_map().values()])