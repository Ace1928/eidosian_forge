from tensorflow.python.framework import tensor_shape
def get_unsharded_shape(self, shapes):
    """Returns the shape of an unsharded Tensor given a list of shards.

    When given a list of shapes of shards, returns the shape of the
    unsharded Tensor that would generate the shards. Sets defaults for the
    policy if number_of_shards or shard_dimension is None.

    Args:
      shapes: The shapes of the Tensor shards to be combined.

    Returns:
      The shape of the unsharded version of the Tensor.

    Raises:
      ValueError: if shapes is not a list of length
        self.number_of_shards; or any element of shapes is not a valid
        shape consistent with the sharding policy; or the list of
        shapes is not a valid sharding of a full shape.
      TypeError: if an element of shapes is not convertible to a
        TensorShape
    """
    self._fill_default_values()
    if len(shapes) != self.number_of_shards:
        raise ValueError(f'Shapes {shapes} is length {len(shapes)} but must be a list of length number_of_shards={self.number_of_shards}')
    unsharded_shapes = [self._unshard_shape(s) for s in shapes]
    for i in range(self.number_of_shards - 1):
        if not unsharded_shapes[i].is_compatible_with(unsharded_shapes[self.number_of_shards - 1]):
            raise ValueError(f'Sharded shapes {shapes} are not consistent shards of a full shape sharded {self.number_of_shards} ways along dimension {self.shard_dimension}.')
    return unsharded_shapes[0]