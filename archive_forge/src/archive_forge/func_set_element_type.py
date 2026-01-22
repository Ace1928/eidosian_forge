from tensorflow.python.util.tf_export import tf_export
def set_element_type(entity, dtype, shape=UNSPECIFIED):
    """Indicates that the entity is expected hold items of specified type/shape.

  The staged TensorFlow ops will reflect and assert this data type. Ignored
  otherwise.

  Args:
    entity: The entity to annotate.
    dtype: TensorFlow dtype value to assert for entity.
    shape: Optional shape to assert for entity.
  """
    del entity
    del dtype
    del shape