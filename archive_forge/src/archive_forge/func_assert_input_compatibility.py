from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import backend
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def assert_input_compatibility(input_spec, inputs, layer_name):
    """Checks compatibility between the layer and provided inputs.

  This checks that the tensor(s) `inputs` verify the input assumptions
  of a layer (if any). If not, a clear and actional exception gets raised.

  Args:
      input_spec: An InputSpec instance, list of InputSpec instances, a nested
          structure of InputSpec instances, or None.
      inputs: Input tensor, list of input tensors, or a nested structure of
          input tensors.
      layer_name: String, name of the layer (for error message formatting).

  Raises:
      ValueError: in case of mismatch between
          the provided inputs and the expectations of the layer.
  """
    if not input_spec:
        return
    input_spec = nest.flatten(input_spec)
    if isinstance(inputs, dict):
        names = [spec.name for spec in input_spec]
        if all(names):
            list_inputs = []
            for name in names:
                if name not in inputs:
                    raise ValueError('Missing data for input "%s". You passed a data dictionary with keys %s. Expected the following keys: %s' % (name, list(inputs.keys()), names))
                list_inputs.append(inputs[name])
            inputs = list_inputs
    inputs = nest.flatten(inputs)
    for x in inputs:
        if not hasattr(x, 'shape'):
            raise TypeError('Inputs to a layer should be tensors. Got: %s' % (x,))
    if len(inputs) != len(input_spec):
        raise ValueError('Layer ' + layer_name + ' expects ' + str(len(input_spec)) + ' input(s), but it received ' + str(len(inputs)) + ' input tensors. Inputs received: ' + str(inputs))
    for input_index, (x, spec) in enumerate(zip(inputs, input_spec)):
        if spec is None:
            continue
        shape = tensor_shape.TensorShape(x.shape)
        if shape.rank is None:
            return
        if spec.ndim is not None and (not spec.allow_last_axis_squeeze):
            ndim = shape.rank
            if ndim != spec.ndim:
                raise ValueError('Input ' + str(input_index) + ' of layer ' + layer_name + ' is incompatible with the layer: expected ndim=' + str(spec.ndim) + ', found ndim=' + str(ndim) + '. Full shape received: ' + str(tuple(shape)))
        if spec.max_ndim is not None:
            ndim = x.shape.rank
            if ndim is not None and ndim > spec.max_ndim:
                raise ValueError('Input ' + str(input_index) + ' of layer ' + layer_name + ' is incompatible with the layer: expected max_ndim=' + str(spec.max_ndim) + ', found ndim=' + str(ndim))
        if spec.min_ndim is not None:
            ndim = x.shape.rank
            if ndim is not None and ndim < spec.min_ndim:
                raise ValueError('Input ' + str(input_index) + ' of layer ' + layer_name + ' is incompatible with the layer: : expected min_ndim=' + str(spec.min_ndim) + ', found ndim=' + str(ndim) + '. Full shape received: ' + str(tuple(shape)))
        if spec.dtype is not None:
            if x.dtype.name != spec.dtype:
                raise ValueError('Input ' + str(input_index) + ' of layer ' + layer_name + ' is incompatible with the layer: expected dtype=' + str(spec.dtype) + ', found dtype=' + str(x.dtype))
        shape_as_list = shape.as_list()
        if spec.axes:
            for axis, value in spec.axes.items():
                if hasattr(value, 'value'):
                    value = value.value
                if value is not None and shape_as_list[int(axis)] not in {value, None}:
                    raise ValueError('Input ' + str(input_index) + ' of layer ' + layer_name + ' is incompatible with the layer: expected axis ' + str(axis) + ' of input shape to have value ' + str(value) + ' but received input with shape ' + display_shape(x.shape))
        if spec.shape is not None and shape.rank is not None:
            spec_shape = spec.shape
            if spec.allow_last_axis_squeeze:
                if shape_as_list and shape_as_list[-1] == 1:
                    shape_as_list = shape_as_list[:-1]
                if spec_shape and spec_shape[-1] == 1:
                    spec_shape = spec_shape[:-1]
            for spec_dim, dim in zip(spec_shape, shape_as_list):
                if spec_dim is not None and dim is not None:
                    if spec_dim != dim:
                        raise ValueError('Input ' + str(input_index) + ' is incompatible with layer ' + layer_name + ': expected shape=' + str(spec.shape) + ', found shape=' + display_shape(x.shape))