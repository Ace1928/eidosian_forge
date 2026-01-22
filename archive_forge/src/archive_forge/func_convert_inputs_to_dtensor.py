from tensorflow.dtensor.python import accelerator_util
from tensorflow.dtensor.python import api as d_api
from tensorflow.dtensor.python import input_util
from tensorflow.dtensor.python import layout
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values as values_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import summary_ops_v2
def convert_inputs_to_dtensor(inputs, mesh):
    """Convert any input types to DTensor instance."""
    if isinstance(inputs, DTensorDistributedValue):
        return inputs.get_dtensor()
    elif isinstance(inputs, values_lib.DistributedValues):
        return convert_per_replica_to_dtensor(inputs, mesh)
    elif isinstance(inputs, input_util._DTensorIterator):
        return inputs
    elif tensor_util.is_tensor(inputs):
        if context.executing_eagerly():
            if d_api.is_dtensor(inputs):
                return inputs
            else:
                _raise_unsupported_input_type_error(inputs)
        else:
            return inputs
    else:
        _raise_unsupported_input_type_error(inputs)