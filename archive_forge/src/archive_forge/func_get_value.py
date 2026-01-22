import abc
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.data.util import structure
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import gen_optional_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def get_value(self, name=None):
    with ops.name_scope(name, 'OptionalGetValue', [self._variant_tensor]) as scope:
        with ops.colocate_with(self._variant_tensor):
            result = gen_optional_ops.optional_get_value(self._variant_tensor, name=scope, output_types=structure.get_flat_tensor_types(self._element_spec), output_shapes=structure.get_flat_tensor_shapes(self._element_spec))
        return structure.from_tensor_list(self._element_spec, result)