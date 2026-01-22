import numpy as np
from tensorflow.python.eager import backprop as backprop_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
@test_util.run_in_graph_and_eager_modes()
def testSecondGradient(self):
    with self.session() as sess:
        labels = constant_op.constant([3, 0, 1], name='labels')
        logits = constant_op.constant([0.3, 0.4, 0.1, 1.2, 0.1, 1.9, 0.1, 0.7, 0.8, 0.2, 1.3, 1.3], shape=[3, 4], dtype=dtypes.float64, name='logits')

        def xent_grad(logits):
            with backprop_lib.GradientTape() as tape:
                tape.watch(logits)
                return tape.gradient(nn_ops.sparse_softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits, name='xent'), [logits])[0]
        analytical, numerical = gradient_checker_v2.compute_gradient(xent_grad, [logits])
        if not context.executing_eagerly() and (not config.is_op_determinism_enabled()):
            op_names = [op.op_def.name for op in sess.graph.get_operations() if op.op_def]
            self.assertIn('BatchMatMulV2', op_names)
    tol = 5e-08
    self.assertAllClose(analytical, numerical, atol=tol, rtol=tol)