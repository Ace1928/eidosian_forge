import queue
from tensorflow.python.framework import dtypes
def obtain_all_variant_tensor_ops(dataset):
    """Given an input dataset, finds all dataset ops used for construction.

  A series of transformations would have created this dataset with each
  transformation including zero or more Dataset ops, each producing a dataset
  variant tensor. This method outputs all of them.

  Args:
    dataset: Dataset to find variant tensors for.

  Returns:
    A list of variant_tensor producing dataset ops used to construct this
    dataset.
  """
    return _traverse(dataset, lambda op: op.outputs[0].dtype == dtypes.variant)