from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import step_fn
from tensorflow.python.distribute import strategy_test_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.layers import core
from tensorflow.python.layers import normalization
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
def minimize_loss_example(optimizer, use_bias=False, use_callable_loss=True):
    """Example of non-distribution-aware legacy code."""

    def dataset_fn():
        dataset = dataset_ops.Dataset.from_tensors([[1.0]]).repeat()
        return dataset.batch(1, drop_remainder=True)
    layer = core.Dense(1, use_bias=use_bias)

    def model_fn(x):
        """A very simple model written by the user."""

        def loss_fn():
            y = array_ops.reshape(layer(x), []) - constant_op.constant(1.0)
            return y * y
        if strategy_test_lib.is_optimizer_v2_instance(optimizer):
            return optimizer.minimize(loss_fn, lambda: layer.trainable_variables)
        elif use_callable_loss:
            return optimizer.minimize(loss_fn)
        else:
            return optimizer.minimize(loss_fn())
    return (model_fn, dataset_fn, layer)