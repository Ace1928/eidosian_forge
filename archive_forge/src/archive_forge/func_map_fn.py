import collections
import csv
import functools
import gzip
import numpy as np
from tensorflow.python import tf2
from tensorflow.python.data.experimental.ops import error_ops
from tensorflow.python.data.experimental.ops import parsing_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import map_op
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops import readers as core_readers
from tensorflow.python.data.util import convert
from tensorflow.python.data.util import nest
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util.tf_export import tf_export
def map_fn(*columns):
    """Organizes columns into a features dictionary.

    Args:
      *columns: list of `Tensor`s corresponding to one csv record.
    Returns:
      An OrderedDict of feature names to values for that particular record. If
      label_name is provided, extracts the label feature to be returned as the
      second element of the tuple.
    """
    features = collections.OrderedDict(zip(column_names, columns))
    if label_name is not None:
        label = features.pop(label_name)
        return (features, label)
    return features