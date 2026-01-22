import string
from typing import List, Sequence
from networkx import MultiDiGraph
import pennylane as qml
from pennylane import numpy as pnp
from .utils import MeasureNode, PrepareNode
def qcut_processing_fn_sample(results: Sequence, communication_graph: MultiDiGraph, shots: int) -> List:
    """
    Function to postprocess samples for the :func:`cut_circuit_mc() <pennylane.cut_circuit_mc>`
    transform. This removes superfluous mid-circuit measurement samples from fragment
    circuit outputs.

    .. note::

        This function is designed for use as part of the sampling-based circuit cutting workflow.
        Check out the :func:`qml.cut_circuit_mc() <pennylane.cut_circuit_mc>` transform for more details.

    Args:
        results (Sequence): a collection of sample-based execution results generated from the
            random expansion of circuit fragments over measurement and preparation node configurations
        communication_graph (nx.MultiDiGraph): the communication graph determining connectivity
            between circuit fragments
        shots (int): the number of shots

    Returns:
        List[tensor_like]: the sampled output for all terminal measurements over the number of shots given
    """
    results = _reshape_results(results, shots)
    res0 = results[0][0]
    out_degrees = [d for _, d in communication_graph.out_degree]
    samples = []
    for result in results:
        sample = []
        for fragment_result, out_degree in zip(result, out_degrees):
            sample.append(fragment_result[:-out_degree or None])
        samples.append(pnp.hstack(sample))
    return [qml.math.convert_like(pnp.array(samples), res0)]