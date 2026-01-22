import functools
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_case
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def resize_fn(images_t, new_size):
    """Resize core function, passed to _resize_images_common."""
    scale_and_translate_methods = [ResizeMethod.LANCZOS3, ResizeMethod.LANCZOS5, ResizeMethod.GAUSSIAN, ResizeMethod.MITCHELLCUBIC]

    def resize_with_scale_and_translate(method):
        scale = math_ops.cast(new_size, dtype=dtypes.float32) / math_ops.cast(array_ops.shape(images_t)[1:3], dtype=dtypes.float32)
        return gen_image_ops.scale_and_translate(images_t, new_size, scale, array_ops.zeros([2]), kernel_type=method, antialias=antialias)
    if method == ResizeMethod.BILINEAR:
        if antialias:
            return resize_with_scale_and_translate('triangle')
        else:
            return gen_image_ops.resize_bilinear(images_t, new_size, half_pixel_centers=True)
    elif method == ResizeMethod.NEAREST_NEIGHBOR:
        return gen_image_ops.resize_nearest_neighbor(images_t, new_size, half_pixel_centers=True)
    elif method == ResizeMethod.BICUBIC:
        if antialias:
            return resize_with_scale_and_translate('keyscubic')
        else:
            return gen_image_ops.resize_bicubic(images_t, new_size, half_pixel_centers=True)
    elif method == ResizeMethod.AREA:
        return gen_image_ops.resize_area(images_t, new_size)
    elif method in scale_and_translate_methods:
        return resize_with_scale_and_translate(method)
    else:
        raise ValueError('Resize method is not implemented: {}'.format(method))