import re
import string
import numpy as np
from keras.src import activations
from keras.src import backend
from keras.src import constraints
from keras.src import dtype_policies
from keras.src import initializers
from keras.src import ops
from keras.src import quantizers
from keras.src import regularizers
from keras.src.api_export import keras_export
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
def quantized_build(self, input_shape, mode):
    shape_data = _analyze_einsum_string(self.equation, self.bias_axes, input_shape, self.partial_output_shape)
    kernel_shape, _, _ = shape_data
    if mode == 'int8':
        self._input_reduced_axes, self._kernel_reduced_axes, self._input_transpose_axes, self._kernel_transpose_axes, self._input_expand_axes, self._kernel_expand_axes, self._input_squeeze_axes, self._kernel_squeeze_axes, self._custom_gradient_equation, self._kernel_reverse_transpose_axes = _analyze_quantization_info(self.equation, self.input_spec.ndim)
        self.inputs_quantizer = quantizers.AbsMaxQuantizer(axis=-1)
        self._kernel = self.add_weight(name='kernel', shape=kernel_shape, initializer='zeros', dtype='int8', trainable=False)
        kernel_scale_shape = np.array(kernel_shape)
        kernel_scale_shape[self._kernel_reduced_axes] = 1
        kernel_scale_shape = kernel_scale_shape[self._kernel_transpose_axes]
        kernel_scale_shape = kernel_scale_shape.tolist()
        for a in sorted(self._kernel_expand_axes):
            kernel_scale_shape.insert(a, 1)
        for a in sorted(self._kernel_squeeze_axes, reverse=True):
            kernel_scale_shape.pop(a)
        self.kernel_scale = self.add_weight(name='kernel_scale', shape=kernel_scale_shape, initializer='ones', trainable=False)