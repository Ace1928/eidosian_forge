from tensorflow.python.framework import tensor_shape
def set_shard_dimension(self, shard_dimension):
    """Sets the shard dimension for the current policy.

    If the policy has been frozen then shard_dimension must match the
    existing setting.

    Args:
      shard_dimension: The shard dimension to use in the policy.

    Raises:
      ValueError: If the policy has been frozen and shard_dimension
        differs from the frozen value, or shard_dimension can't be
        interpreted as a Dimension.
    """
    if self._frozen:
        if self._shard_dimension != shard_dimension:
            raise ValueError("Can't set shard dimension to %d since it has been frozen to use %d." % (shard_dimension, self._shard_dimension))
    else:
        self._shard_dimension = tensor_shape.as_dimension(shard_dimension)