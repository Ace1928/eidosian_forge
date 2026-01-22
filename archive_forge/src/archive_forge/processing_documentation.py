import string
from typing import List, Sequence
from networkx import MultiDiGraph
import pennylane as qml
from pennylane import numpy as pnp
from .utils import MeasureNode, PrepareNode
Process a flat list of execution results from all circuit fragments into the corresponding
    tensors.

    This function slices ``results`` according to the expected size of fragment tensors derived from
    the ``prepare_nodes`` and ``measure_nodes`` and then passes onto ``_process_tensor`` for further
    transformation.

    Args:
        results (tensor_like): A collection of execution results, provided as a flat tensor,
            corresponding to the expansion of circuit fragments in the communication graph over
            measurement and preparation node configurations. These results are processed into
            tensors by this function.
        prepare_nodes (Sequence[Sequence[PrepareNode]]): a sequence whose length is equal to the
            number of circuit fragments, with each element used here to determine the number of
            preparation nodes in a given fragment
        measure_nodes (Sequence[Sequence[MeasureNode]]): a sequence whose length is equal to the
            number of circuit fragments, with each element used here to determine the number of
            measurement nodes in a given fragment

    Returns:
        List[tensor_like]: the tensors for each circuit fragment in the communication graph
    