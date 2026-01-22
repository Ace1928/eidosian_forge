import numpy as np
from qiskit.exceptions import QiskitError
def format_level_1_memory(memory):
    """Format an experiment result memory object for measurement level 1.

    Args:
        memory (list): Memory from experiment with `meas_level==1`. `avg` or
            `single` will be inferred from shape of result memory.

    Returns:
        np.ndarray: Measurement level 1 complex numpy array

    Raises:
        QiskitError: If the returned numpy array does not have 1 (avg) or 2 (single)
            indices.
    """
    formatted_memory = _list_to_complex_array(memory)
    if not 1 <= len(formatted_memory.shape) <= 2:
        raise QiskitError('Level one memory is not of correct shape.')
    return formatted_memory