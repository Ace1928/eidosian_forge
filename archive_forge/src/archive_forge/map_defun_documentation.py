from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import gen_dataset_ops
Map a function on the list of tensors unpacked from `elems` on dimension 0.

  Args:
    fn: A function (`function.defun`) that takes a list of tensors and returns
      another list of tensors. The output list has the same types as
      output_dtypes. The elements of the output list have the same dimension 0
      as `elems`, and the remaining dimensions correspond to those of
      `fn_output_shapes`.
    elems: A list of tensors.
    output_dtypes: A list of dtypes corresponding to the output types of the
      function.
    output_shapes: A list of `TensorShape`s corresponding to the output shapes
      from each invocation of the function on slices of inputs.
    max_intra_op_parallelism: An integer. If positive, sets the max parallelism
      limit of each function call to this.

  Raises:
    ValueError: if any of the inputs are malformed.

  Returns:
    A list of `Tensor` objects with the same types as `output_dtypes`.
  