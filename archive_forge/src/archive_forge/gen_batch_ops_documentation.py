import collections
from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.security.fuzzing.py import annotation_types as _atypes
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export
from typing import TypeVar, List
Gradient of Unbatch.

  Acts like Batch but using the given batch_index index of batching things as they
  become available. This ensures that the gradients are propagated back in the
  same session which did the forward pass.

  original_input: The input to the Unbatch operation this is the gradient of.
  batch_index: The batch_index given to the Unbatch operation this is the gradient
  of.
  grad: The downstream gradient.
  id: The id scalar emitted by Batch.
  batched_grad: The return value, either an empty tensor or the batched gradient.
  container: Container to control resource sharing.
  shared_name: Instances of UnbatchGrad with the same container and shared_name
   are assumed to possibly belong to the same batch. If left empty, the op name
   will be used as the shared name.

  Args:
    original_input: A `Tensor`.
    batch_index: A `Tensor` of type `int64`.
    grad: A `Tensor`. Must have the same type as `original_input`.
    id: A `Tensor` of type `int64`.
    container: An optional `string`. Defaults to `""`.
    shared_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `original_input`.
  