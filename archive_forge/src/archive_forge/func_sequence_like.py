import collections as _collections
import enum
import typing
from typing import Protocol
import six as _six
import wrapt as _wrapt
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util.compat import collections_abc as _collections_abc
def sequence_like(instance, args):
    """Converts the sequence `args` to the same type as `instance`.

  Args:
    instance: an instance of `tuple`, `list`, `namedtuple`, `dict`,
      `collections.OrderedDict`, or `composite_tensor.Composite_Tensor` or
      `type_spec.TypeSpec`.
    args: items to be converted to the `instance` type.

  Returns:
    `args` with the type of `instance`.
  """
    if _is_mutable_mapping(instance):
        result = dict(zip(_tf_core_sorted(instance), args))
        instance_type = type(instance)
        if instance_type == _collections.defaultdict:
            d = _collections.defaultdict(instance.default_factory)
        else:
            d = instance_type()
        for key in instance:
            d[key] = result[key]
        return d
    elif _is_mapping(instance):
        result = dict(zip(_tf_core_sorted(instance), args))
        instance_type = type(instance)
        if not getattr(instance_type, '__supported_by_tf_nest__', False):
            tf_logging.log_first_n(tf_logging.WARN, 'Mapping types may not work well with tf.nest. Prefer using MutableMapping for {}'.format(instance_type), 1)
        try:
            return instance_type(((key, result[key]) for key in instance))
        except TypeError as err:
            raise TypeError('Error creating an object of type {} like {}. Note that it must accept a single positional argument representing an iterable of key-value pairs, in addition to self. Cause: {}'.format(type(instance), instance, err))
    elif _is_mapping_view(instance):
        return list(args)
    elif is_namedtuple(instance) or _is_attrs(instance):
        if isinstance(instance, _wrapt.ObjectProxy):
            instance_type = type(instance.__wrapped__)
        else:
            instance_type = type(instance)
        return instance_type(*args)
    elif _is_composite_tensor(instance):
        assert len(args) == 1
        spec = instance._type_spec
        return spec._from_components(args[0])
    elif _is_type_spec(instance):
        assert len(args) == 1
        return instance._from_components(args[0])
    elif isinstance(instance, _six.moves.range):
        return sequence_like(list(instance), args)
    elif isinstance(instance, _wrapt.ObjectProxy):
        return type(instance)(sequence_like(instance.__wrapped__, args))
    elif isinstance(instance, CustomNestProtocol):
        metadata = instance.__tf_flatten__()[0]
        return instance.__tf_unflatten__(metadata, tuple(args))
    else:
        return type(instance)(args)