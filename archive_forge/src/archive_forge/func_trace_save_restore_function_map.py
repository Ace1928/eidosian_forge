from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import nest
def trace_save_restore_function_map(obj, factory_data_list):
    """Traces all save and restore functions in the provided factory list.

  Args:
    obj: `Trackable` object.
    factory_data_list: List of `_CheckpointFactoryData`.

  Returns:
    Dict mapping atttribute names to tuples of concrete save/restore functions.
  """
    saveable_fns = {}
    for factory_data in factory_data_list:
        saveable_factory = factory_data.factory
        attribute_name = factory_data.name
        if resource_variable_ops.is_resource_variable(obj) or resource_variable_ops.is_resource_variable(saveable_factory) or (not callable(saveable_factory)):
            continue
        concrete_save, concrete_restore = _trace_save_restore_functions(saveable_factory, obj)
        if not concrete_save:
            continue
        saveable_fns[attribute_name] = (concrete_save, concrete_restore)
    return saveable_fns