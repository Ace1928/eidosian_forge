import collections
import contextlib
import functools
import itertools
import threading
import numpy as np
from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.optimizer_v2 import adadelta as adadelta_v2
from tensorflow.python.keras.optimizer_v2 import adagrad as adagrad_v2
from tensorflow.python.keras.optimizer_v2 import adam as adam_v2
from tensorflow.python.keras.optimizer_v2 import adamax as adamax_v2
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_v2
from tensorflow.python.keras.optimizer_v2 import nadam as nadam_v2
from tensorflow.python.keras.optimizer_v2 import rmsprop as rmsprop_v2
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.util import tf_decorator
def get_multi_io_model(branch_a, branch_b, shared_input_branch=None, shared_output_branch=None):
    """Builds a multi-io model that contains two branches.

  The produced model will be of the type specified by `get_model_type`.

  To build a two-input, two-output model:
    Specify a list of layers for branch a and branch b, but do not specify any
    shared input branch or shared output branch. The resulting model will apply
    each branch to a different input, to produce two outputs.

    The first value in branch_a must be the Keras 'Input' layer for branch a,
    and the first value in branch_b must be the Keras 'Input' layer for
    branch b.

    example usage:
    ```
    branch_a = [Input(shape=(2,), name='a'), Dense(), Dense()]
    branch_b = [Input(shape=(3,), name='b'), Dense(), Dense()]

    model = get_multi_io_model(branch_a, branch_b)
    ```

  To build a two-input, one-output model:
    Specify a list of layers for branch a and branch b, and specify a
    shared output branch. The resulting model will apply
    each branch to a different input. It will then apply the shared output
    branch to a tuple containing the intermediate outputs of each branch,
    to produce a single output. The first layer in the shared_output_branch
    must be able to merge a tuple of two tensors.

    The first value in branch_a must be the Keras 'Input' layer for branch a,
    and the first value in branch_b must be the Keras 'Input' layer for
    branch b.

    example usage:
    ```
    input_branch_a = [Input(shape=(2,), name='a'), Dense(), Dense()]
    input_branch_b = [Input(shape=(3,), name='b'), Dense(), Dense()]
    shared_output_branch = [Concatenate(), Dense(), Dense()]

    model = get_multi_io_model(input_branch_a, input_branch_b,
                               shared_output_branch=shared_output_branch)
    ```
  To build a one-input, two-output model:
    Specify a list of layers for branch a and branch b, and specify a
    shared input branch. The resulting model will take one input, and apply
    the shared input branch to it. It will then respectively apply each branch
    to that intermediate result in parallel, to produce two outputs.

    The first value in the shared_input_branch must be the Keras 'Input' layer
    for the whole model. Branch a and branch b should not contain any Input
    layers.

    example usage:
    ```
    shared_input_branch = [Input(shape=(2,), name='in'), Dense(), Dense()]
    output_branch_a = [Dense(), Dense()]
    output_branch_b = [Dense(), Dense()]


    model = get_multi_io_model(output__branch_a, output_branch_b,
                               shared_input_branch=shared_input_branch)
    ```

  Args:
    branch_a: A sequence of layers for branch a of the model.
    branch_b: A sequence of layers for branch b of the model.
    shared_input_branch: An optional sequence of layers to apply to a single
      input, before applying both branches to that intermediate result. If set,
      the model will take only one input instead of two. Defaults to None.
    shared_output_branch: An optional sequence of layers to merge the
      intermediate results produced by branch a and branch b. If set,
      the model will produce only one output instead of two. Defaults to None.

  Returns:
    A multi-io model of the type specified by `get_model_type`, specified
    by the different branches.
  """
    if shared_input_branch:
        inputs = shared_input_branch[0]
        shared_input_branch = shared_input_branch[1:]
    else:
        inputs = (branch_a[0], branch_b[0])
        branch_a = branch_a[1:]
        branch_b = branch_b[1:]
    model_type = get_model_type()
    if model_type == 'subclass':
        return _MultiIOSubclassModel(branch_a, branch_b, shared_input_branch, shared_output_branch)
    if model_type == 'subclass_custom_build':
        return _MultiIOSubclassModelCustomBuild(lambda: branch_a, lambda: branch_b, lambda: shared_input_branch, lambda: shared_output_branch)
    if model_type == 'sequential':
        raise ValueError('Cannot use `get_multi_io_model` to construct sequential models')
    if model_type == 'functional':
        if shared_input_branch:
            a_and_b = inputs
            for layer in shared_input_branch:
                a_and_b = layer(a_and_b)
            a = a_and_b
            b = a_and_b
        else:
            a, b = inputs
        for layer in branch_a:
            a = layer(a)
        for layer in branch_b:
            b = layer(b)
        outputs = (a, b)
        if shared_output_branch:
            for layer in shared_output_branch:
                outputs = layer(outputs)
        return models.Model(inputs, outputs)
    raise ValueError('Unknown model type {}'.format(model_type))