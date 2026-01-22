import numpy as np
from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl
@staticmethod
def norm_l2(x):
    """L2 normalization"""
    xn = np.square(x).sum(axis=1)
    np.sqrt(xn, out=xn)
    norm = np.maximum(xn.reshape((x.shape[0], -1)), 1e-30)
    return x / norm