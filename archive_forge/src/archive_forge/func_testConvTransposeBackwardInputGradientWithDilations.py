import collections
import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test
@test_util.run_cuda_only
def testConvTransposeBackwardInputGradientWithDilations(self):
    self.testConvTransposeBackwardInputGradient(rate=2)