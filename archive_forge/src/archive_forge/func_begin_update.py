import copy
from typing import Any, cast
import srsly
from ..compat import mxnet as mx
from ..optimizers import Optimizer
from ..types import ArgsKwargs, FloatsXd
from ..util import (
from .shim import Shim
def begin_update(self, inputs: ArgsKwargs):
    """Pass the inputs through to the underlying MXNet model, keeping
        track of which items in the input are tensors requiring gradients.
        If the model returns a single value, it is converted into a one-element
        tuple. Return the outputs and a callback to backpropagate.
        """
    mx.autograd.set_training(train_mode=True)
    mx.autograd.set_recording(True)
    output = self._model(*inputs.args, **inputs.kwargs)

    def backprop(grads):
        mx.autograd.set_recording(False)
        mx.autograd.backward(*grads.args, **grads.kwargs)
        return convert_recursive(lambda x: hasattr(x, 'grad'), lambda x: x.grad, inputs)
    return (output, backprop)