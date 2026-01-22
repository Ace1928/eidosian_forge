import inspect
import tensorflow.compat.v2 as tf
from keras.src.dtensor import dtensor_api as dtensor
def running_with_dtensor_strategy():
    """Check whether running with a `Strategy` that is backed by DTensor.

    In the DTensor based training, all the tensors are in global context, which
    is different from the local context. Some keras components need to
    behave differently, e.g. BatchNormalization and SyncBatchNormalization, as
    well as optimizers.

    This check will help those layer to branch the logic and keep the correct
    behavior between different context.
    """
    if not tf.distribute.has_strategy():
        return False
    strategy = tf.distribute.get_strategy()
    return getattr(strategy, '_mesh', None) is not None