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
def recreate_saveable_objects(saveable_fn_by_name, temp_session):
    """Returns a dict of SaveableObject factories generated from loaded fns."""
    names_and_slices = []
    with ops.init_scope():
        for save_fn, _ in saveable_fn_by_name.values():
            for tensor_info in save_fn(''):
                name = tensor_info['name']
                slice_spec = tensor_info['slice_spec']
                if not context.executing_eagerly():
                    sess = ops.get_default_session()
                    if sess is None:
                        if temp_session[0] is not None:
                            sess = temp_session[0]
                        else:
                            sess = temp_session[0] = session.Session()
                    name, slice_spec = sess.run([name, slice_spec])
                names_and_slices.append((_convert_to_string(name), _convert_to_string(slice_spec)))
    saveable_factories = {}
    for name, (save_fn, restore_fn) in saveable_fn_by_name.items():
        saveable_factories[name] = functools.partial(RestoredSaveableObject, names_and_slices=names_and_slices, save_function=save_fn, restore_function=restore_fn)
    return saveable_factories