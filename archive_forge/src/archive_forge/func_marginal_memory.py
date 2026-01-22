from typing import Sequence, Union, Optional, Dict, List
from collections import Counter
from copy import deepcopy
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.result.result import Result
from qiskit.result.counts import Counts
from qiskit.result.distributions.probability import ProbDistribution
from qiskit.result.distributions.quasi import QuasiDistribution
from qiskit.result.postprocess import _bin_to_hex
from qiskit._accelerate import results as results_rs  # pylint: disable=no-name-in-module
def marginal_memory(memory: Union[List[str], np.ndarray], indices: Optional[List[int]]=None, int_return: bool=False, hex_return: bool=False, avg_data: bool=False, parallel_threshold: int=1000) -> Union[List[str], np.ndarray]:
    """Marginalize shot memory

    This function is multithreaded and will launch a thread pool with threads equal to the number
    of CPUs by default. You can tune the number of threads with the ``RAYON_NUM_THREADS``
    environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would limit the thread pool
    to 4 threads.

    Args:
        memory: The input memory list, this is either a list of hexadecimal strings to be marginalized
            representing measure level 2 memory or a numpy array representing level 0 measurement
            memory (single or avg) or level 1 measurement memory (single or avg).
        indices: The bit positions of interest to marginalize over. If
            ``None`` (default), do not marginalize at all.
        int_return: If set to ``True`` the output will be a list of integers.
            By default the return type is a bit string. This and ``hex_return``
            are mutually exclusive and can not be specified at the same time. This option only has an
            effect with memory level 2.
        hex_return: If set to ``True`` the output will be a list of hexadecimal
            strings. By default the return type is a bit string. This and
            ``int_return`` are mutually exclusive and can not be specified
            at the same time. This option only has an effect with memory level 2.
        avg_data: If a 2 dimensional numpy array is passed in for ``memory`` this can be set to
            ``True`` to indicate it's a avg level 0 data instead of level 1
            single data.
        parallel_threshold: The number of elements in ``memory`` to start running in multiple
            threads. If ``len(memory)`` is >= this value, the function will run in multiple
            threads. By default this is set to 1000.

    Returns:
        marginal_memory: The list of marginalized memory

    Raises:
        ValueError: if both ``int_return`` and ``hex_return`` are set to ``True``
    """
    if int_return and hex_return:
        raise ValueError('Either int_return or hex_return can be specified but not both')
    if isinstance(memory, np.ndarray):
        if int_return:
            raise ValueError('int_return option only works with memory list input')
        if hex_return:
            raise ValueError('hex_return option only works with memory list input')
        if indices is None:
            return memory.copy()
        if memory.ndim == 1:
            return results_rs.marginal_measure_level_1_avg(memory, indices)
        if memory.ndim == 2:
            if avg_data:
                return results_rs.marginal_measure_level_0_avg(memory, indices)
            else:
                return results_rs.marginal_measure_level_1(memory, indices)
        if memory.ndim == 3:
            return results_rs.marginal_measure_level_0(memory, indices)
        raise ValueError('Invalid input memory array')
    return results_rs.marginal_memory(memory, indices, return_int=int_return, return_hex=hex_return, parallel_threshold=parallel_threshold)