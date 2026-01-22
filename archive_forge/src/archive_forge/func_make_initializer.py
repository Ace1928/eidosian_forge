import abc
import threading
import warnings
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.data.ops import iterator_autograph
from tensorflow.python.data.ops import optional_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_utils
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training.saver import BaseSaverBuilder
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def make_initializer(self, dataset, name=None):
    """Returns a `tf.Operation` that initializes this iterator on `dataset`.

    Args:
      dataset: A `Dataset` whose `element_spec` if compatible with this
        iterator.
      name: (Optional.) A name for the created operation.

    Returns:
      A `tf.Operation` that can be run to initialize this iterator on the given
      `dataset`.

    Raises:
      TypeError: If `dataset` and this iterator do not have a compatible
        `element_spec`.
    """
    with ops.name_scope(name, 'make_initializer') as name:
        dataset_output_types = nest.map_structure(lambda component_spec: component_spec._to_legacy_output_types(), dataset.element_spec)
        dataset_output_shapes = nest.map_structure(lambda component_spec: component_spec._to_legacy_output_shapes(), dataset.element_spec)
        dataset_output_classes = nest.map_structure(lambda component_spec: component_spec._to_legacy_output_classes(), dataset.element_spec)
        nest.assert_same_structure(self.output_types, dataset_output_types)
        nest.assert_same_structure(self.output_shapes, dataset_output_shapes)
        for iterator_class, dataset_class in zip(nest.flatten(self.output_classes), nest.flatten(dataset_output_classes)):
            if iterator_class is not dataset_class:
                raise TypeError(f'Expected output classes {self.output_classes!r} but got dataset with output classes {dataset_output_classes!r}.')
        for iterator_dtype, dataset_dtype in zip(nest.flatten(self.output_types), nest.flatten(dataset_output_types)):
            if iterator_dtype != dataset_dtype:
                raise TypeError(f'Expected output types {self.output_types!r} but got dataset with output types {dataset_output_types!r}.')
        for iterator_shape, dataset_shape in zip(nest.flatten(self.output_shapes), nest.flatten(dataset_output_shapes)):
            if not iterator_shape.is_compatible_with(dataset_shape):
                raise TypeError(f'Expected output shapes compatible with {self.output_shapes!r} but got dataset with output shapes {dataset_output_shapes!r}.')
    with ops.colocate_with(self._iterator_resource):
        return gen_dataset_ops.make_iterator(dataset._variant_tensor, self._iterator_resource, name=name)