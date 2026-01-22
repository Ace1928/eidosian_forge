import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_nonmaxsuppression_single_box() -> None:
    node = onnx.helper.make_node('NonMaxSuppression', inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'], outputs=['selected_indices'])
    boxes = np.array([[[0.0, 0.0, 1.0, 1.0]]]).astype(np.float32)
    scores = np.array([[[0.9]]]).astype(np.float32)
    max_output_boxes_per_class = np.array([3]).astype(np.int64)
    iou_threshold = np.array([0.5]).astype(np.float32)
    score_threshold = np.array([0.0]).astype(np.float32)
    selected_indices = np.array([[0, 0, 0]]).astype(np.int64)
    expect(node, inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold], outputs=[selected_indices], name='test_nonmaxsuppression_single_box')