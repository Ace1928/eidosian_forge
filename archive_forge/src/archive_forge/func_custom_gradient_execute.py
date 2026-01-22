import inspect
import logging
import warnings
import tensorflow as tf
from tensorflow.python.eager import context
import pennylane as qml
from pennylane.measurements import Shots
@tf.custom_gradient
def custom_gradient_execute(*parameters):
    """An execution of tapes with VJP's registered with tensorflow.

        Args:
            *parameters (TensorLike): the trainable parameters for the tapes.

        Closure:
            tapes (tuple(QuantumTape)): the tapes to execute. Contains tensorflow parameters.
            numpy_tapes (tuple(QuantumTape)): tapes but with numpy parameters
            numpy_params (list(numpy.ndarray)): numpy versions of ``parameters``.
            jpc (JacobianProductCalculator): a class that can calculate the VJP.

        Returns:
            ResultBatch, Callable: the result of executing the tapes and a function capable of calculating the VJP.

        """

    def vjp_fn(*dy, **tfkwargs):
        dy = _recursive_conj(dy)
        if not differentiable:
            inner_tapes = numpy_tapes
        elif not context.executing_eagerly():
            warnings.warn('PennyLane does not provide the higher order derivatives of tensorflow jacobians.')
            inner_tapes = set_parameters_on_copy(tapes, numpy_params)
        else:
            inner_tapes = tapes
        dy_dtype = dy[0].dtype
        nested_dy = _res_restructured(dy, tapes)
        try:
            vjps = jpc.compute_vjp(inner_tapes, nested_dy)
        except AttributeError as e:
            message = 'device VJPs cannot be vectorized with tensorflow. To use device_vjp=True, \n set experimental_use_pfor=False as a keyword argument to GradientTape.jacobian\n and set persistent=True to GradientTape.'
            raise ValueError(message) from e
        vjps = _to_tensors(vjps, dtype=dy_dtype)
        if isinstance(vjps, tuple):
            extended_vjps = []
            for vjp in vjps:
                if vjp is not None and 0 not in qml.math.shape(vjp):
                    extended_vjps.extend(qml.math.unstack(vjp))
            vjps = tuple(extended_vjps)
        variables = tfkwargs.get('variables')
        return (vjps, variables) if variables is not None else vjps
    return (res, vjp_fn)