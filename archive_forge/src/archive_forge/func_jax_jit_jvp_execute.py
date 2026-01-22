from functools import partial
import jax
import jax.numpy as jnp
import pennylane as qml
from pennylane.typing import ResultBatch
from ..jacobian_products import _compute_jvps
from .jax import _NonPytreeWrapper
def jax_jit_jvp_execute(tapes, execute_fn, jpc, device):
    """Execute a batch of tapes with JAX parameters using JVP derivatives.

    Args:
        tapes (Sequence[.QuantumTape]): batch of tapes to execute
        execute_fn (Callable[[Sequence[.QuantumTape]], ResultBatch]): a function that turns a batch of circuits into results
        jpc (JacobianProductCalculator): a class that can compute the Jacobian for the input tapes.
        device (pennylane.Device, pennylane.devices.Device): The device used for execution. Used to determine the shapes of outputs for
            pure callback calls.

    Returns:
        TensorLike: A nested tuple of tape results. Each element in
        the returned tuple corresponds in order to the provided tapes.

    """
    if any((m.return_type in (qml.measurements.Counts, qml.measurements.AllCounts) for t in tapes for m in t.measurements)):
        raise NotImplementedError("The JAX-JIT interface doesn't support qml.counts.")
    parameters = tuple((tuple(t.get_parameters(trainable_only=False)) for t in tapes))
    return _execute_jvp_jit(parameters, _NonPytreeWrapper(tuple(tapes)), execute_fn, jpc, device)