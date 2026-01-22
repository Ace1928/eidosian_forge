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
@tf_export('image.central_crop')
@dispatch.add_dispatch_support
def central_crop(image, central_fraction):
    """Crop the central region of the image(s).

  Remove the outer parts of an image but retain the central region of the image
  along each dimension. If we specify `central_fraction = 0.5`, this function
  returns the region marked with "X" in the below diagram. The larger the value
  of `central_fraction`, the larger the dimension of the region to be cropped
  and retained.

       --------
      |        |
      |  XXXX  |
      |  XXXX  |
      |        |   where "X" is the central 50% of the image.
       --------

  This function works on either a single image (`image` is a 3-D Tensor), or a
  batch of images (`image` is a 4-D Tensor).

  Usage Example:

  >>> x = [[[1.0, 2.0, 3.0],
  ...       [4.0, 5.0, 6.0],
  ...       [7.0, 8.0, 9.0],
  ...       [10.0, 11.0, 12.0]],
  ...     [[13.0, 14.0, 15.0],
  ...       [16.0, 17.0, 18.0],
  ...       [19.0, 20.0, 21.0],
  ...       [22.0, 23.0, 24.0]],
  ...     [[25.0, 26.0, 27.0],
  ...       [28.0, 29.0, 30.0],
  ...       [31.0, 32.0, 33.0],
  ...       [34.0, 35.0, 36.0]],
  ...     [[37.0, 38.0, 39.0],
  ...       [40.0, 41.0, 42.0],
  ...       [43.0, 44.0, 45.0],
  ...       [46.0, 47.0, 48.0]]]
  >>> tf.image.central_crop(x, 0.5)
  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=
  array([[[16., 17., 18.],
          [19., 20., 21.]],
         [[28., 29., 30.],
          [31., 32., 33.]]], dtype=float32)>

  Args:
    image: Either a 3-D float Tensor of shape [height, width, depth], or a 4-D
      Tensor of shape [batch_size, height, width, depth].
    central_fraction: float (0, 1], fraction of size to crop

  Raises:
    ValueError: if central_crop_fraction is not within (0, 1].

  Returns:
    3-D / 4-D float Tensor, as per the input.
  """
    with ops.name_scope(None, 'central_crop', [image]):
        image = ops.convert_to_tensor(image, name='image')
        central_fraction_static = tensor_util.constant_value(central_fraction)
        if central_fraction_static is not None:
            if central_fraction_static <= 0.0 or central_fraction_static > 1.0:
                raise ValueError('central_fraction must be within (0, 1]')
            if central_fraction_static == 1.0:
                return image
        else:
            assert_ops = _assert(math_ops.logical_or(central_fraction > 0.0, central_fraction <= 1.0), ValueError, 'central_fraction must be within (0, 1]')
            image = control_flow_ops.with_dependencies(assert_ops, image)
        _AssertAtLeast3DImage(image)
        rank = image.get_shape().ndims
        if rank != 3 and rank != 4:
            raise ValueError('`image` should either be a Tensor with rank = 3 or rank = 4. Had rank = {}.'.format(rank))

        def _get_dim(tensor, idx):
            static_shape = tensor.get_shape().dims[idx].value
            if static_shape is not None:
                return (static_shape, False)
            return (array_ops.shape(tensor)[idx], True)
        if rank == 3:
            img_h, dynamic_h = _get_dim(image, 0)
            img_w, dynamic_w = _get_dim(image, 1)
            img_d = image.get_shape()[2]
        else:
            img_bs = image.get_shape()[0]
            img_h, dynamic_h = _get_dim(image, 1)
            img_w, dynamic_w = _get_dim(image, 2)
            img_d = image.get_shape()[3]
        dynamic_h = dynamic_h or central_fraction_static is None
        dynamic_w = dynamic_w or central_fraction_static is None
        if dynamic_h:
            img_hd = math_ops.cast(img_h, dtypes.float64)
            bbox_h_start = math_ops.cast((img_hd - img_hd * math_ops.cast(central_fraction, dtypes.float64)) / 2, dtypes.int32)
        else:
            img_hd = float(img_h)
            bbox_h_start = int((img_hd - img_hd * central_fraction_static) / 2)
        if dynamic_w:
            img_wd = math_ops.cast(img_w, dtypes.float64)
            bbox_w_start = math_ops.cast((img_wd - img_wd * math_ops.cast(central_fraction, dtypes.float64)) / 2, dtypes.int32)
        else:
            img_wd = float(img_w)
            bbox_w_start = int((img_wd - img_wd * central_fraction_static) / 2)
        bbox_h_size = img_h - bbox_h_start * 2
        bbox_w_size = img_w - bbox_w_start * 2
        if rank == 3:
            bbox_begin = array_ops_stack.stack([bbox_h_start, bbox_w_start, 0])
            bbox_size = array_ops_stack.stack([bbox_h_size, bbox_w_size, -1])
        else:
            bbox_begin = array_ops_stack.stack([0, bbox_h_start, bbox_w_start, 0])
            bbox_size = array_ops_stack.stack([-1, bbox_h_size, bbox_w_size, -1])
        image = array_ops.slice(image, bbox_begin, bbox_size)
        if rank == 3:
            image.set_shape([None if dynamic_h else bbox_h_size, None if dynamic_w else bbox_w_size, img_d])
        else:
            image.set_shape([img_bs, None if dynamic_h else bbox_h_size, None if dynamic_w else bbox_w_size, img_d])
        return image