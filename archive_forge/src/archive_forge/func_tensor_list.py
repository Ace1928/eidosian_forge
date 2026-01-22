from tensorflow.python.autograph.operators import data_structures
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import tensor_util
def tensor_list(elements, element_dtype=None, element_shape=None, use_tensor_array=False):
    """Creates an tensor list and populates it with the given elements.

  This function provides a more uniform access to tensor lists and tensor
  arrays, and allows optional initialization.

  Note: this function is a simplified wrapper. If you need greater control,
  it is recommended to use the underlying implementation directly.

  Args:
    elements: Iterable[tf.Tensor, ...], the elements to initially fill the list
        with
    element_dtype: Optional[tf.DType], data type for the elements in the list;
        required if the list is empty
    element_shape: Optional[tf.TensorShape], shape for the elements in the list;
        required if the list is empty
    use_tensor_array: bool, whether to use the more compatible but restrictive
        tf.TensorArray implementation
  Returns:
    Union[tf.Tensor, tf.TensorArray], the new list.
  Raises:
    ValueError: for invalid arguments
  """
    _validate_list_constructor(elements, element_dtype, element_shape)
    if use_tensor_array:
        return data_structures.tf_tensor_array_new(elements, element_dtype, element_shape)
    else:
        return data_structures.tf_tensor_list_new(elements, element_dtype, element_shape)