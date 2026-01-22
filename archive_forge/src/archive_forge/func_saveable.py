from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.ops import gen_lookup_ops
from tensorflow.python.training import saver as saver_module
@property
def saveable(self):
    if context.executing_eagerly():
        return CheckpointedOp.CustomSaveable(self, self.name)
    else:
        return self._saveable