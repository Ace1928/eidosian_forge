import functools
import inspect
import os
import warnings
import pennylane as qml
def tape_fn(self, obj, *args, **kwargs):
    """The tape transform function.

        This is the function that is called if a datastructure is passed
        that contains multiple operations.

        Args:
            obj (pennylane.QNode, .QuantumTape, or Callable): A quantum node, tape,
                or function that applies quantum operations.
            *args: positional arguments to pass to the function
            **kwargs: keyword arguments to pass to the function

        Returns:
            any: the result of evaluating the transform

        Raises:
            .OperationTransformError: if no tape transform function is defined

        .. seealso:: :meth:`.op_transform.tape_transform`
        """
    if self._tape_fn is None:
        raise OperationTransformError('This transform does not support tapes or QNodes with multiple operations.')
    return self._tape_fn(obj, *args, **kwargs)