import numpy as np
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@property
def flat_values(self):
    """The innermost `values` array for this ragged tensor value."""
    rt_values = self.values
    while isinstance(rt_values, RaggedTensorValue):
        rt_values = rt_values.values
    return rt_values