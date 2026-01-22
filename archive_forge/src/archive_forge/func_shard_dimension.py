from tensorflow.python.framework import tensor_shape
@property
def shard_dimension(self):
    """Returns the shard dimension of the policy or None if unspecified."""
    return self._shard_dimension