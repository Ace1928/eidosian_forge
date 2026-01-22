from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import nest
@def_function.function(input_signature=[tensor_spec.TensorSpec([], dtypes.string)])
def save_fn(checkpoint_key):
    maybe_saveable = saveable_factory(name=checkpoint_key)
    if isinstance(maybe_saveable, saveable_object.SaveableObject):
        maybe_saveable = [maybe_saveable]
    saveables[:] = maybe_saveable
    ret = []
    for saveable in saveables:
        for spec in saveable.specs:
            ret.append({'name': spec.name, 'tensor': spec.tensor, 'slice_spec': spec.slice_spec})
    return ret