import copy
import functools
import inspect
import os
import types
import warnings
from typing import Callable, Tuple
import pennylane as qml
from pennylane.typing import ResultBatch
def custom_qnode_wrapper(self, fn):
    """Register a custom QNode execution wrapper function
        for the batch transform.

        **Example**

        .. code-block:: python

            def my_transform(tape, *targs, **tkwargs):
                ...
                return tapes, processing_fn

            @my_transform.custom_qnode_wrapper
            def my_custom_qnode_wrapper(self, qnode, targs, tkwargs):
                def wrapper_fn(*args, **kwargs):
                    # construct QNode
                    qnode.construct(args, kwargs)
                    # apply transform to QNode's tapes
                    tapes, processing_fn = self.construct(qnode.qtape, *targs, **tkwargs)
                    # execute tapes and return processed result
                    ...
                    return processing_fn(results)
                return wrapper_fn

        The custom QNode execution wrapper must have arguments
        ``self`` (the batch transform object), ``qnode`` (the input QNode
        to transform and execute), ``targs`` and ``tkwargs`` (the transform
        arguments and keyword arguments respectively).

        It should return a callable object that accepts the *same* arguments
        as the QNode, and returns the transformed numerical result.

        The default :meth:`~.default_qnode_wrapper` method may be called
        if only pre- or post-processing dependent on QNode arguments is required:

        .. code-block:: python

            @my_transform.custom_qnode_wrapper
            def my_custom_qnode_wrapper(self, qnode, targs, tkwargs):
                transformed_qnode = self.default_qnode_wrapper(qnode)

                def wrapper_fn(*args, **kwargs):
                    args, kwargs = pre_process(args, kwargs)
                    res = transformed_qnode(*args, **kwargs)
                    ...
                    return ...
                return wrapper_fn
        """
    self.qnode_wrapper = types.MethodType(fn, self)