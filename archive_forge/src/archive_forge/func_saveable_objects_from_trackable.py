import functools
from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import ref_variable
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable import python_state
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.types import core
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.tracking.saveable_objects_from_trackable', v1=[])
def saveable_objects_from_trackable(obj, tf1_saver=False):
    """Returns SaveableObject factory dict from a Trackable.

  Args:
    obj: A `Trackable`
    tf1_saver: Boolean, whether this is being called from a TF1 Saver (
        `tf.compat.v1.train.Saver`). When this is True, the SaveableObject will
        be generated from `obj`'s legacy `_gather_saveables_for_checkpoint` fn.
        When saving with TF2, `Trackable._serialize_from_tensors` is preferred.

  Returns:
    A dict mapping attribute names to SaveableObject factories (callables that
    produce a SaveableObject).
  """
    if isinstance(obj, python_state.PythonState):
        return {python_state.PYTHON_STATE: functools.partial(_PythonStringStateSaveable, state_callback=obj.serialize, restore_callback=obj.deserialize)}
    if tf1_saver:
        saveable_factories = obj._gather_saveables_for_checkpoint()
        if saveable_factories:
            return saveable_factories
    if trackable_has_serialize_to_tensor(obj):

        def create_saveable(name='', call_with_mapped_captures=None):
            save_fn = obj._serialize_to_tensors
            if call_with_mapped_captures and isinstance(save_fn, core.ConcreteFunction):
                tensor_dict = call_with_mapped_captures(save_fn, [])
            else:
                tensor_dict = save_fn()
            specs = []
            local_names = []
            for tensor_name, maybe_tensor in tensor_dict.items():
                local_names.append(tensor_name)
                if not isinstance(maybe_tensor, dict):
                    maybe_tensor = {'': maybe_tensor}
                spec_name = name + trackable_utils.escape_local_name(tensor_name)
                for slice_spec, tensor in maybe_tensor.items():
                    if isinstance(tensor, saveable_object.SaveSpec):
                        spec = tensor
                        spec.name = spec_name
                        spec.slice_spec = slice_spec
                    else:
                        spec = saveable_object.SaveSpec(tensor, slice_spec, spec_name)
                    specs.append(spec)
            return TrackableSaveable(obj=obj, specs=specs, name=name, local_names=local_names, prefix=saveable_compat.get_saveable_name(obj) or '', call_with_mapped_captures=call_with_mapped_captures)
        return {trackable_utils.SERIALIZE_TO_TENSORS_NAME: create_saveable}
    else:
        return obj._gather_saveables_for_checkpoint()