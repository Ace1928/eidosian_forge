import tensorflow as tf
from ray.util.annotations import PublicAPI
A utility function that overrides default config for Tensorflow Dataset.

    This should be used on a TensorFlow ``Dataset`` created by calling
    ``iter_tf_batches()`` on a ``ray.data.Dataset`` returned by
    ``ray.train.get_dataset_shard()`` since the dataset has already
    been sharded across the workers.

    Args:
        tf_dataset_shard (tf.data.Dataset): A TensorFlow Dataset.

    Returns:
        A TensorFlow Dataset with:
            - autosharding turned off
            - prefetching turned on with autotune enabled
    