import functools
import os
import copy
import warnings
import types
from typing import Sequence
import pennylane as qml
from pennylane.typing import ResultBatch
def custom_qnode_transform(self, fn):
    """Register a custom QNode execution wrapper function
        for the batch transform.

        **Example**

        .. code-block:: python

            @transform
            def my_transform(tape, *targs, **tkwargs):
                ...
                return tapes, processing_fn

            @my_transform.custom_qnode_transform
            def my_custom_qnode_wrapper(self, qnode, targs, tkwargs):
                tkwargs = {**tkwargs, shots=100}
                return self.default_qnode_transform(qnode, targs, tkwargs)

        The custom QNode execution wrapper must have arguments
        ``self`` (the batch transform object), ``qnode`` (the input QNode
        to transform and execute), ``targs`` and ``tkwargs`` (the transform
        arguments and keyword arguments respectively).

        It should return a QNode that accepts the *same* arguments as the
        input QNode with the transform applied.

        The default :meth:`~.default_qnode_transform` method may be called
        if only pre- or post-processing dependent on QNode arguments is required.
        """
    self._qnode_transform = types.MethodType(fn, self)