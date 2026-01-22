from collections import abc
import contextlib
import threading
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import tpu_values as tpu_values_lib
from tensorflow.python.distribute import values as values_lib
from tensorflow.python.distribute.reduce_util import ReduceOp
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def regroup(values, wrap_class=values_lib.PerReplica, always_wrap=False):
    """Makes a nest per-replica into a nest of PerReplica/Mirrored values.

  Args:
    values: Values to regroup
    wrap_class: Class that `values` be wrapped in.
    always_wrap: Always wrap the `values` in `wrap_class` even if the values
        are the same except for DistributeVariable.
  Returns:
    Wrapped `values`.
  """
    v0 = values[0]
    if isinstance(v0, list):
        for v in values[1:]:
            assert isinstance(v, list)
            assert len(v) == len(v0), 'len(v) == %d, len(v0) == %d, v: %s, v0: %s' % (len(v), len(v0), v, v0)
        return [regroup(tuple((v[i] for v in values)), wrap_class, always_wrap) for i in range(len(v0))]
    if isinstance(v0, tuple):
        for v in values[1:]:
            assert isinstance(v, tuple)
            assert len(v) == len(v0), f'Values to regroup had different lengths: len(v) == {len(v)}, len(v0) == {len(v0)}, v: {v}, v0: {v0}'
        regrouped_tuple = tuple((regroup(tuple((v[i] for v in values)), wrap_class, always_wrap) for i in range(len(v0))))
        if hasattr(v0, '_fields'):
            assert hasattr(v0, '_make')
            return v0._make(regrouped_tuple)
        else:
            return regrouped_tuple
    if isinstance(v0, abc.Mapping):
        v0keys = v0.keys()
        for v in values[1:]:
            assert isinstance(v, abc.Mapping), 'v[0]: %r  v[i]: %r' % (v0, v)
            assert set(v.keys()) == set(v0keys), 'v[0].keys: %s  v[i].keys: %s' % (set(v0keys), set(v.keys()))
        return type(v0)({key: regroup(tuple((v[key] for v in values)), wrap_class, always_wrap) for key in v0keys})
    same_id = True
    for v in values[1:]:
        if v is not v0:
            same_id = False
            break
    if same_id and isinstance(v0, values_lib.DistributedVariable):
        return v0
    if same_id and (not always_wrap) and (value_container(v0) is v0):
        return v0
    if not isinstance(v0, resource_variable_ops._UnreadVariable) and value_container(v0) is not v0:
        assert not isinstance(v0, values_lib.MirroredVariable), 'ids = %s, values = %s' % ([id(v) for v in values], values)
        distributed_container = value_container(v0)
        assert distributed_container is not None
        for v in values[1:]:
            assert distributed_container is value_container(v)
        return distributed_container
    return wrap_class(values)