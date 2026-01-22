import logging
from typing import Tuple, Callable
import dataclasses
import jax
import jax.numpy as jnp
import pennylane as qml
from pennylane.transforms import convert_to_numpy_parameters
from pennylane.typing import ResultBatch
def get_jax_interface_name(tapes):
    """Check all parameters in each tape and output the name of the suitable
    JAX interface.

    This function checks each tape and determines if any of the gate parameters
    was transformed by a JAX transform such as ``jax.jit``. If so, it outputs
    the name of the JAX interface with jit support.

    Note that determining if jit support should be turned on is done by
    checking if parameters are abstract. Parameters can be abstract not just
    for ``jax.jit``, but for other JAX transforms (vmap, pmap, etc.) too. The
    reason is that JAX doesn't have a public API for checking whether or not
    the execution is within the jit transform.

    Args:
        tapes (Sequence[.QuantumTape]): batch of tapes to execute

    Returns:
        str: name of JAX interface that fits the tape parameters, "jax" or
        "jax-jit"
    """
    for t in tapes:
        for op in t:
            op = op.obs if hasattr(op, 'obs') else op
            if op is not None:
                for param in op.data:
                    if qml.math.is_abstract(param):
                        return 'jax-jit'
    return 'jax'