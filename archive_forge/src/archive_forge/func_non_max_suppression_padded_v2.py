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
@def_function.function(experimental_implements='non_max_suppression_padded_v2')
def non_max_suppression_padded_v2(boxes, scores, max_output_size, iou_threshold=0.5, score_threshold=float('-inf'), sorted_input=False, canonicalized_coordinates=False, tile_size=512):
    """Non-maximum suppression.

  Prunes away boxes that have high intersection-over-union (IOU) overlap
  with previously selected boxes. Bounding boxes are supplied as
  `[y1, x1, y2, x2]`, where `(y1, x1)` and `(y2, x2)` are the coordinates of any
  diagonal pair of box corners and the coordinates can be provided as normalized
  (i.e., lying in the interval `[0, 1]`) or absolute. The bounding box
  coordinates are cannonicalized to `[y_min, x_min, y_max, x_max]`,
  where `(y_min, x_min)` and `(y_max, x_mas)` are the coordinates of the lower
  left and upper right corner. User may indiciate the input box coordinates are
  already canonicalized to eliminate redundant work by setting
  canonicalized_coordinates to `True`. Note that this algorithm is agnostic to
  where the origin is in the coordinate system. Note that this algorithm is
  invariant to orthogonal transformations and translations of the coordinate
  system; thus translating or reflections of the coordinate system result in the
  same boxes being selected by the algorithm.

  Similar to tf.image.non_max_suppression, non_max_suppression_padded
  implements hard NMS but can operate on a batch of images and improves
  performance by titling the bounding boxes. Non_max_suppression_padded should
  be preferred over tf.image_non_max_suppression when running on devices with
  abundant parallelsim for higher computation speed. For soft NMS, refer to
  tf.image.non_max_suppression_with_scores.

  While a serial NMS algorithm iteratively uses the highest-scored unprocessed
  box to suppress boxes, this algorithm uses many boxes to suppress other boxes
  in parallel. The key idea is to partition boxes into tiles based on their
  score and suppresses boxes tile by tile, thus achieving parallelism within a
  tile. The tile size determines the degree of parallelism.

  In cross suppression (using boxes of tile A to suppress boxes of tile B),
  all boxes in A can independently suppress boxes in B.

  Self suppression (suppressing boxes of the same tile) needs to be iteratively
  applied until there's no more suppression. In each iteration, boxes that
  cannot be suppressed are used to suppress boxes in the same tile.

  boxes = boxes.pad_to_multiply_of(tile_size)
  num_tiles = len(boxes) // tile_size
  output_boxes = []
  for i in range(num_tiles):
    box_tile = boxes[i*tile_size : (i+1)*tile_size]
    for j in range(i - 1):
      # in parallel suppress boxes in box_tile using boxes from suppressing_tile
      suppressing_tile = boxes[j*tile_size : (j+1)*tile_size]
      iou = _bbox_overlap(box_tile, suppressing_tile)
      # if the box is suppressed in iou, clear it to a dot
      box_tile *= _update_boxes(iou)
    # Iteratively handle the diagnal tile.
    iou = _box_overlap(box_tile, box_tile)
    iou_changed = True
    while iou_changed:
      # boxes that are not suppressed by anything else
      suppressing_boxes = _get_suppressing_boxes(iou)
      # boxes that are suppressed by suppressing_boxes
      suppressed_boxes = _get_suppressed_boxes(iou, suppressing_boxes)
      # clear iou to 0 for boxes that are suppressed, as they cannot be used
      # to suppress other boxes any more
      new_iou = _clear_iou(iou, suppressed_boxes)
      iou_changed = (new_iou != iou)
      iou = new_iou
    # remaining boxes that can still suppress others, are selected boxes.
    output_boxes.append(_get_suppressing_boxes(iou))
    if len(output_boxes) >= max_output_size:
      break

  Args:
    boxes: a tensor of rank 2 or higher with a shape of [..., num_boxes, 4].
      Dimensions except the last two are batch dimensions. The last dimension
      represents box coordinates, given as [y_1, x_1, y_2, x_2]. The coordinates
      on each dimension can be given in any order
      (see also `canonicalized_coordinates`) but must describe a box with
      a positive area.
    scores: a tensor of rank 1 or higher with a shape of [..., num_boxes].
    max_output_size: a scalar integer `Tensor` representing the maximum number
      of boxes to be selected by non max suppression.
    iou_threshold: a float representing the threshold for deciding whether boxes
      overlap too much with respect to IoU (intersection over union).
    score_threshold: a float representing the threshold for box scores. Boxes
      with a score that is not larger than this threshold will be suppressed.
    sorted_input: a boolean indicating whether the input boxes and scores
      are sorted in descending order by the score.
    canonicalized_coordinates: if box coordinates are given as
    `[y_min, x_min, y_max, x_max]`, setting to True eliminate redundant
     computation to canonicalize box coordinates.
    tile_size: an integer representing the number of boxes in a tile, i.e.,
      the maximum number of boxes per image that can be used to suppress other
      boxes in parallel; larger tile_size means larger parallelism and
      potentially more redundant work.
  Returns:
    idx: a tensor with a shape of [..., num_boxes] representing the
      indices selected by non-max suppression. The leading dimensions
      are the batch dimensions of the input boxes. All numbers are within
      [0, num_boxes). For each image (i.e., idx[i]), only the first num_valid[i]
      indices (i.e., idx[i][:num_valid[i]]) are valid.
    num_valid: a tensor of rank 0 or higher with a shape of [...]
      representing the number of valid indices in idx. Its dimensions are the
      batch dimensions of the input boxes.
   Raises:
    ValueError: When set pad_to_max_output_size to False for batched input.
  """

    def _sort_scores_and_boxes(scores, boxes):
        """Sort boxes based their score from highest to lowest.

    Args:
      scores: a tensor with a shape of [batch_size, num_boxes] representing
        the scores of boxes.
      boxes: a tensor with a shape of [batch_size, num_boxes, 4] representing
        the boxes.
    Returns:
      sorted_scores: a tensor with a shape of [batch_size, num_boxes]
        representing the sorted scores.
      sorted_boxes: a tensor representing the sorted boxes.
      sorted_scores_indices: a tensor with a shape of [batch_size, num_boxes]
        representing the index of the scores in a sorted descending order.
    """
        with ops.name_scope('sort_scores_and_boxes'):
            sorted_scores_indices = sort_ops.argsort(scores, axis=1, direction='DESCENDING')
            sorted_scores = array_ops.gather(scores, sorted_scores_indices, axis=1, batch_dims=1)
            sorted_boxes = array_ops.gather(boxes, sorted_scores_indices, axis=1, batch_dims=1)
        return (sorted_scores, sorted_boxes, sorted_scores_indices)
    batch_dims = array_ops.shape(boxes)[:-2]
    num_boxes = array_ops.shape(boxes)[-2]
    boxes = array_ops.reshape(boxes, [-1, num_boxes, 4])
    scores = array_ops.reshape(scores, [-1, num_boxes])
    batch_size = array_ops.shape(boxes)[0]
    if score_threshold != float('-inf'):
        with ops.name_scope('filter_by_score'):
            score_mask = math_ops.cast(scores > score_threshold, scores.dtype)
            scores *= score_mask
            box_mask = array_ops.expand_dims(math_ops.cast(score_mask, boxes.dtype), 2)
            boxes *= box_mask
    if not canonicalized_coordinates:
        with ops.name_scope('canonicalize_coordinates'):
            y_1, x_1, y_2, x_2 = array_ops.split(value=boxes, num_or_size_splits=4, axis=2)
            y_1_is_min = math_ops.reduce_all(math_ops.less_equal(y_1[0, 0, 0], y_2[0, 0, 0]))
            y_min, y_max = tf_cond.cond(y_1_is_min, lambda: (y_1, y_2), lambda: (y_2, y_1))
            x_1_is_min = math_ops.reduce_all(math_ops.less_equal(x_1[0, 0, 0], x_2[0, 0, 0]))
            x_min, x_max = tf_cond.cond(x_1_is_min, lambda: (x_1, x_2), lambda: (x_2, x_1))
            boxes = array_ops.concat([y_min, x_min, y_max, x_max], axis=2)
    if not sorted_input:
        scores, boxes, sorted_indices = _sort_scores_and_boxes(scores, boxes)
    else:
        sorted_indices = array_ops.zeros_like(scores, dtype=dtypes.int32)
    pad = math_ops.cast(math_ops.ceil(math_ops.cast(math_ops.maximum(num_boxes, max_output_size), dtypes.float32) / math_ops.cast(tile_size, dtypes.float32)), dtypes.int32) * tile_size - num_boxes
    boxes = array_ops.pad(math_ops.cast(boxes, dtypes.float32), [[0, 0], [0, pad], [0, 0]])
    scores = array_ops.pad(math_ops.cast(scores, dtypes.float32), [[0, 0], [0, pad]])
    num_boxes_after_padding = num_boxes + pad
    num_iterations = num_boxes_after_padding // tile_size

    def _loop_cond(unused_boxes, unused_threshold, output_size, idx):
        return math_ops.logical_and(math_ops.reduce_min(output_size) < max_output_size, idx < num_iterations)

    def suppression_loop_body(boxes, iou_threshold, output_size, idx):
        return _suppression_loop_body(boxes, iou_threshold, output_size, idx, tile_size)
    selected_boxes, _, output_size, _ = while_loop.while_loop(_loop_cond, suppression_loop_body, [boxes, iou_threshold, array_ops.zeros([batch_size], dtypes.int32), constant_op.constant(0)], shape_invariants=[tensor_shape.TensorShape([None, None, 4]), tensor_shape.TensorShape([]), tensor_shape.TensorShape([None]), tensor_shape.TensorShape([])])
    num_valid = math_ops.minimum(output_size, max_output_size)
    idx = num_boxes_after_padding - math_ops.cast(nn_ops.top_k(math_ops.cast(math_ops.reduce_any(selected_boxes > 0, [2]), dtypes.int32) * array_ops.expand_dims(math_ops.range(num_boxes_after_padding, 0, -1), 0), max_output_size)[0], dtypes.int32)
    idx = math_ops.minimum(idx, num_boxes - 1)
    if not sorted_input:
        index_offsets = math_ops.range(batch_size) * num_boxes
        gather_idx = array_ops.reshape(idx + array_ops.expand_dims(index_offsets, 1), [-1])
        idx = array_ops.reshape(array_ops.gather(array_ops.reshape(sorted_indices, [-1]), gather_idx), [batch_size, -1])
    invalid_index = array_ops.zeros([batch_size, max_output_size], dtype=dtypes.int32)
    idx_index = array_ops.expand_dims(math_ops.range(max_output_size), 0)
    num_valid_expanded = array_ops.expand_dims(num_valid, 1)
    idx = array_ops.where(idx_index < num_valid_expanded, idx, invalid_index)
    num_valid = array_ops.reshape(num_valid, batch_dims)
    return (idx, num_valid)