import dataclasses
from typing import Optional, Tuple
import numpy as np
from onnx.reference.op_run import OpRun
def prepare_compute(self, pc: PrepareContext, boxes_tensor: np.ndarray, scores_tensor: np.ndarray, max_output_boxes_per_class_tensor: np.ndarray, iou_threshold_tensor: np.ndarray, score_threshold_tensor: np.ndarray):
    pc.boxes_data_ = boxes_tensor
    pc.scores_data_ = scores_tensor
    if max_output_boxes_per_class_tensor.size != 0:
        pc.max_output_boxes_per_class_ = max_output_boxes_per_class_tensor
    if iou_threshold_tensor.size != 0:
        pc.iou_threshold_ = iou_threshold_tensor
    if score_threshold_tensor.size != 0:
        pc.score_threshold_ = score_threshold_tensor
    pc.boxes_size_ = boxes_tensor.size
    pc.scores_size_ = scores_tensor.size
    boxes_dims = boxes_tensor.shape
    scores_dims = scores_tensor.shape
    pc.num_batches_ = boxes_dims[0]
    pc.num_classes_ = scores_dims[1]
    pc.num_boxes_ = boxes_dims[1]