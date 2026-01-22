import collections
import pprint
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.eager import forwardprop_util
from tensorflow.python.eager import record
from tensorflow.python.eager.graph_only_ops import graph_placeholder
from tensorflow.python.eager.polymorphic_function import atomic_function
from tensorflow.python.eager.polymorphic_function import attributes as attributes_lib
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.eager.polymorphic_function import saved_model_exported_concrete
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import core
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
def pretty_printed_signature(self, verbose=True):
    """Returns a string summarizing the signature of this concrete function."""
    if not verbose:
        return self._structured_signature_summary(default_values=True)

    def pretty_print_spec(spec):
        """Returns a string describing the spec for a single argument."""
        if isinstance(spec, tensor_lib.TensorSpec):
            return '{} Tensor, shape={}'.format(spec.dtype.name, spec.shape)
        elif nest.is_nested(spec):
            pieces = nest.flatten(spec, expand_composites=False)
            markers = [_Marker('<{}>'.format(i + 1)) for i in range(len(pieces))]
            structure = nest.pack_sequence_as(spec, markers)
            result = pprint.pformat(structure, width=10000)
            for marker, piece in zip(markers, pieces):
                result += '\n      {}: {}'.format(marker, pretty_print_spec(piece))
            return result
        else:
            return repr(spec)
    lines = [self._structured_signature_summary(default_values=True)]
    arg_specs, kwarg_specs = self.structured_input_signature
    names = function_type_utils.to_arg_names(self.function_type)
    arg_details = []
    for name, spec in zip(names[:len(arg_specs)], list(arg_specs)):
        if _contains_type_spec(spec):
            arg_details.append('    {}: {}'.format(name, pretty_print_spec(spec)))
    if kwarg_specs:
        for kwarg in sorted(kwarg_specs):
            spec = kwarg_specs[kwarg]
            if _contains_type_spec(spec):
                arg_details.append('    {}: {}'.format(kwarg, pretty_print_spec(spec)))
    if arg_details:
        lines.append('  Args:')
        lines.extend(arg_details)
    lines.append('  Returns:')

    def spec_from_value(value):
        if isinstance(value, type_spec.TypeSpec):
            return value
        return type_spec.type_spec_from_value(value)
    lines.append('    {}'.format(pretty_print_spec(nest.map_structure(spec_from_value, self.structured_outputs))))
    return '\n'.join(lines)