import collections
import copy
import math
import re
from typing import Optional
from tensorflow.core.protobuf.tpu import optimization_parameters_pb2
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2 as elc
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu_system_metadata as tpu_system_metadata_lib
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.util.tf_export import tf_export
def get_enqueue_datas_list_from_ragged_tensors_list(rg_tensors_list):
    """Convenient function for generate_enqueue_ops().

  Args:
    rg_tensors_list: a list of dictionary mapping from string of feature names
      to RaggedTensor. Each dictionary is for one TPU core. Dictionaries for the
      same host should be contiguous on the list.

  Returns:
    enqueue_datas_list: a list of dictionary mapping from string
      of feature names to RaggedEnqueueData. Each dictionary is for one
      TPU core. Dictionaries for the same host should be contiguous
      on the list.

  """
    enqueue_datas_list = []
    for rg_tensors in rg_tensors_list:
        enqueue_datas = collections.OrderedDict(((k, RaggedEnqueueData.from_ragged_tensor(v)) for k, v in rg_tensors.items()))
        enqueue_datas_list.append(enqueue_datas)
    return enqueue_datas_list