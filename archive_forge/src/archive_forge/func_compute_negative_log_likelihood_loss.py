import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
def compute_negative_log_likelihood_loss(input, target, weight=None, reduction='mean', ignore_index=None):
    input_shape = input.shape
    if len(input_shape) == 1:
        raise RuntimeError('Unsupported shape')
    target_shape = target.shape
    N = input_shape[0]
    C = input_shape[1]
    gather_weight = None
    if weight is not None:
        gather_weight = np.take(weight, np.array(target, dtype=np.int32), mode='clip')
        if ignore_index is not None:
            gather_weight = np.where(target == ignore_index, 0, gather_weight).astype(dtype=np.float32)
    elif ignore_index is not None:
        gather_weight = np.where(target == ignore_index, 0, 1).astype(dtype=np.float32)
    if len(input_shape) != 3:
        input = input.reshape((N, C, -1))
        target = target.reshape((N, -1))
    D = input.shape[2]
    neg_gather_element_input = np.zeros((N, D), dtype=np.float32)
    for i in range(N):
        for d in range(D):
            if target[i][d] != ignore_index:
                neg_gather_element_input[i][d] = -input[i][target[i][d]][d]
    loss = neg_gather_element_input
    if len(input_shape) != 3:
        loss = loss.reshape(target_shape)
    if gather_weight is not None:
        loss = gather_weight * loss
        if reduction == 'mean':
            loss = loss.sum() / gather_weight.sum()
            return loss
    if reduction == 'mean':
        loss = np.mean(loss)
    elif reduction == 'sum':
        loss = np.sum(loss)
    return loss