import string
from typing import List, Sequence
from networkx import MultiDiGraph
import pennylane as qml
from pennylane import numpy as pnp
from .utils import MeasureNode, PrepareNode
def qcut_processing_fn_mc(results: Sequence, communication_graph: MultiDiGraph, settings: pnp.ndarray, shots: int, classical_processing_fn: callable):
    """
    Function to postprocess samples for the :func:`cut_circuit_mc() <pennylane.cut_circuit_mc>`
    transform. This takes a user-specified classical function to act on bitstrings and
    generates an expectation value.

    .. note::

        This function is designed for use as part of the sampling-based circuit cutting workflow.
        Check out the :func:`qml.cut_circuit_mc() <pennylane.cut_circuit_mc>` transform for more details.

    Args:
        results (Sequence): a collection of sample-based execution results generated from the
            random expansion of circuit fragments over measurement and preparation node configurations
        communication_graph (nx.MultiDiGraph): the communication graph determining connectivity
            between circuit fragments
        settings (np.ndarray): Each element is one of 8 unique values that tracks the specific
            measurement and preparation operations over all configurations. The number of rows is determined
            by the number of cuts and the number of columns is determined by the number of shots.
        shots (int): the number of shots
        classical_processing_fn (callable): A classical postprocessing function to be applied to
            the reconstructed bitstrings. The expected input is a bitstring; a flat array of length ``wires``
            and the output should be a single number within the interval :math:`[-1, 1]`.

    Returns:
        float or tensor_like: the expectation value calculated in accordance to Eq. (35) of
        `Peng et al. <https://arxiv.org/abs/1904.00102>`__
    """
    results = _reshape_results(results, shots)
    res0 = results[0][0]
    out_degrees = [d for _, d in communication_graph.out_degree]
    evals = (0.5, 0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5)
    expvals = []
    for result, setting in zip(results, settings.T):
        sample_terminal = []
        sample_mid = []
        for fragment_result, out_degree in zip(result, out_degrees):
            sample_terminal.append(fragment_result[:-out_degree or None])
            sample_mid.append(fragment_result[-out_degree or len(fragment_result):])
        sample_terminal = pnp.hstack(sample_terminal)
        sample_mid = pnp.hstack(sample_mid)
        assert set(sample_terminal).issubset({pnp.array(0), pnp.array(1)})
        assert set(sample_mid).issubset({pnp.array(-1), pnp.array(1)})
        f = classical_processing_fn(sample_terminal)
        if not -1 <= f <= 1:
            raise ValueError('The classical processing function supplied must give output in the interval [-1, 1]')
        sigma_s = pnp.prod(sample_mid)
        t_s = f * sigma_s
        c_s = pnp.prod([evals[s] for s in setting])
        K = len(sample_mid)
        expvals.append(8 ** K * c_s * t_s)
    return qml.math.convert_like(pnp.mean(expvals), res0)