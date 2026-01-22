from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import nest
@def_function.function(input_signature=restored_type_specs)
def restore_fn(*restored_tensors):
    structured_restored_tensors = nest.pack_sequence_as(tensor_structure, restored_tensors)
    for saveable, restored_tensors in zip(saveables, structured_restored_tensors):
        saveable.restore(restored_tensors, restored_shapes=None)
    return 1