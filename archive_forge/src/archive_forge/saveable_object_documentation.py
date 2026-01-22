Restores this object from 'restored_tensors'.

    Args:
      restored_tensors: the tensors that were loaded from a checkpoint
      restored_shapes: the shapes this object should conform to after
        restore, or None.

    Returns:
      An operation that restores the state of the object.

    Raises:
      ValueError: If the object cannot be restored using the provided
        parameters.
    