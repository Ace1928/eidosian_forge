import itertools
from typing import Optional, Tuple
import numpy as np
from onnx.reference.op_run import OpRun
from onnx.reference.ops._op_common_indices import _get_index, _get_indices
def loop_range():
    return [range(int(round_fct(float(x_shape[i + 2] + pad_shape[i] - kernel_shape[i]) / float(strides_shape[i]) + 1))) for i in range(spatial_size)]