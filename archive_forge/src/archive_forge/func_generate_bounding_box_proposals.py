import collections
from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.security.fuzzing.py import annotation_types as _atypes
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export
from typing import TypeVar, List
def generate_bounding_box_proposals(scores: _atypes.TensorFuzzingAnnotation[_atypes.Float32], bbox_deltas: _atypes.TensorFuzzingAnnotation[_atypes.Float32], image_info: _atypes.TensorFuzzingAnnotation[_atypes.Float32], anchors: _atypes.TensorFuzzingAnnotation[_atypes.Float32], nms_threshold: _atypes.TensorFuzzingAnnotation[_atypes.Float32], pre_nms_topn: _atypes.TensorFuzzingAnnotation[_atypes.Int32], min_size: _atypes.TensorFuzzingAnnotation[_atypes.Float32], post_nms_topn: int=300, name=None):
    """This op produces Region of Interests from given bounding boxes(bbox_deltas) encoded wrt anchors according to eq.2 in arXiv:1506.01497

        The op selects top `pre_nms_topn` scoring boxes, decodes them with respect to anchors,
        applies non-maximal suppression on overlapping boxes with higher than
        `nms_threshold` intersection-over-union (iou) value, discarding boxes where shorter
        side is less than `min_size`.
        Inputs:
        `scores`: A 4D tensor of shape [Batch, Height, Width, Num Anchors] containing the scores per anchor at given position
        `bbox_deltas`: is a tensor of shape [Batch, Height, Width, 4 x Num Anchors] boxes encoded to each anchor
        `anchors`: A 1D tensor of shape [4 x Num Anchors], representing the anchors.
        Outputs:
        `rois`: output RoIs, a 3D tensor of shape [Batch, post_nms_topn, 4], padded by 0 if less than post_nms_topn candidates found.
        `roi_probabilities`: probability scores of each roi in 'rois', a 2D tensor of shape [Batch,post_nms_topn], padded with 0 if needed, sorted by scores.

  Args:
    scores: A `Tensor` of type `float32`.
      A 4-D float tensor of shape `[num_images, height, width, num_achors]` containing scores of the boxes for given anchors, can be unsorted.
    bbox_deltas: A `Tensor` of type `float32`.
      A 4-D float tensor of shape `[num_images, height, width, 4 x num_anchors]`. encoding boxes with respec to each anchor.
      Coordinates are given in the form [dy, dx, dh, dw].
    image_info: A `Tensor` of type `float32`.
      A 2-D float tensor of shape `[num_images, 5]` containing image information Height, Width, Scale.
    anchors: A `Tensor` of type `float32`.
      A 2-D float tensor of shape `[num_anchors, 4]` describing the anchor boxes. Boxes are formatted in the form [y1, x1, y2, x2].
    nms_threshold: A `Tensor` of type `float32`.
      A scalar float tensor for non-maximal-suppression threshold.
    pre_nms_topn: A `Tensor` of type `int32`.
      A scalar int tensor for the number of top scoring boxes to be used as input.
    min_size: A `Tensor` of type `float32`.
      A scalar float tensor. Any box that has a smaller size than min_size will be discarded.
    post_nms_topn: An optional `int`. Defaults to `300`.
      An integer. Maximum number of rois in the output.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (rois, roi_probabilities).

    rois: A `Tensor` of type `float32`.
    roi_probabilities: A `Tensor` of type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'GenerateBoundingBoxProposals', name, scores, bbox_deltas, image_info, anchors, nms_threshold, pre_nms_topn, min_size, 'post_nms_topn', post_nms_topn)
            _result = _GenerateBoundingBoxProposalsOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return generate_bounding_box_proposals_eager_fallback(scores, bbox_deltas, image_info, anchors, nms_threshold, pre_nms_topn, min_size, post_nms_topn=post_nms_topn, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if post_nms_topn is None:
        post_nms_topn = 300
    post_nms_topn = _execute.make_int(post_nms_topn, 'post_nms_topn')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('GenerateBoundingBoxProposals', scores=scores, bbox_deltas=bbox_deltas, image_info=image_info, anchors=anchors, nms_threshold=nms_threshold, pre_nms_topn=pre_nms_topn, min_size=min_size, post_nms_topn=post_nms_topn, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('post_nms_topn', _op._get_attr_int('post_nms_topn'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('GenerateBoundingBoxProposals', _inputs_flat, _attrs, _result)
    _result = _GenerateBoundingBoxProposalsOutput._make(_result)
    return _result