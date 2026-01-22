import numpy as np
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.util import nest
def one_host_numpy_dataset(numpy_input, colocate_with, session):
    """Create a dataset on `colocate_with` from `numpy_input`."""

    def create_colocated_variable(next_creator, **kwargs):
        kwargs['colocate_with'] = colocate_with
        return next_creator(**kwargs)
    numpy_flat = nest.flatten(numpy_input)
    with variable_scope.variable_creator_scope(create_colocated_variable):
        vars_flat = tuple((variable_v1.VariableV1(array_ops.zeros(i.shape, i.dtype), trainable=False) for i in numpy_flat))
    for v, i in zip(vars_flat, numpy_flat):
        init_var_from_numpy(v, i, session)
    vars_nested = nest.pack_sequence_as(numpy_input, vars_flat)
    return dataset_ops.Dataset.from_tensor_slices(vars_nested)