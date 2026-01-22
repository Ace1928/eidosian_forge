import numpy as np
from onnx.reference.ops.aionnxml._common_classifier import (
from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl
from onnx.reference.ops.aionnxml.op_svm_helper import SVMCommon
def write_scores(n_classes, scores, post_transform, add_second_class):
    if n_classes >= 2:
        if post_transform == 'PROBIT':
            res = [compute_probit(score) for score in scores]
            return np.array(res, dtype=scores.dtype)
        if post_transform == 'LOGISTIC':
            return logistic(scores)
        if post_transform == 'SOFTMAX':
            return softmax(scores)
        if post_transform == 'SOFTMAX_ZERO':
            return compute_softmax_zero(scores)
        return scores
    if n_classes == 1:
        if post_transform == 'PROBIT':
            return np.array([compute_probit(scores[0])], dtype=scores.dtype)
        if add_second_class in (0, 1):
            return np.array([1 - scores[0], scores[0]], dtype=scores.dtype)
        if add_second_class in (2, 3):
            if post_transform == 'LOGISTIC':
                return np.array([logistic(-scores[0]), logistic(scores[0])], dtype=scores.dtype)
            if post_transform == 'SOFTMAX':
                return softmax(np.array([-scores[0], scores[0]], dtype=scores.dtype))
            if post_transform == 'SOFTMAX_ZERO':
                return softmax_zero(np.array([-scores[0], scores[0]], dtype=scores.dtype))
            if post_transform == 'PROBIT':
                raise RuntimeError(f'post_transform={post_transform!r} not applicable here.')
            return np.array([-scores[0], scores[0]], dtype=scores.dtype)
        return np.array([scores[0]], dtype=scores.dtype)
    raise NotImplementedError(f'n_classes={n_classes} not supported.')