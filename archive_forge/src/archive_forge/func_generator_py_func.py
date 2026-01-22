import numpy as np
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import script_ops
def generator_py_func(iterator_id):
    """A `py_func` that will be called to invoke the iterator."""
    values = next(generator_state.get_iterator(iterator_id.numpy()))
    try:
        values = structure.normalize_element(values, output_signature)
    except (TypeError, ValueError) as e:
        raise TypeError(f'`generator` yielded an element that did not match the expected structure. The expected structure was {output_signature}, but the yielded element was {values}.') from e
    values_spec = structure.type_spec_from_value(values)
    if not structure.are_compatible(values_spec, output_signature):
        raise TypeError(f'`generator` yielded an element of {values_spec} where an element of {output_signature} was expected.')
    return structure.to_tensor_list(output_signature, values)