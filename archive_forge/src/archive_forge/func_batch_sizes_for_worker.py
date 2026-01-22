from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops.options import ExternalStatePolicy
from tensorflow.python.data.util import nest
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.types import data as data_types
from tensorflow.python.util.tf_export import tf_export
def batch_sizes_for_worker(global_batch_size, num_workers, num_replicas_per_worker, worker_index):
    """Determines how to rebatch a dataset for the given worker.

  Given the global batch size, number of workers, number of replicas per worker,
  and worker index, returns the correct batch sizes for rebatching a dataset
  on worker `worker_index` of `num_workers`, such that each global step (across
  all workers and replicas) will consume global_batch_size elements. The
  returned value should be passed as the `batch_sizes` input parameter to
  `tf.data.experimental.rebatch()`. The returned batch sizes meet the following
  constraints:

  Let G = global_batch_size, W = num_workers, R = num_replicas_per_worker
  (A) for any worker, len(batch_sizes) = W * R
  (B) for any worker, sum(batch_sizes) == G
  (C) for any global step (i.e. R iterations on each worker), the sum of batches
      consumed by replicas across all workers is G.
  (D) any two batch sizes of any two replicas differs by at most one.

  For example, suppose we have G = 7, W = 2, R = 2, and suppose we have two
  files which each contain 7 elements:

  ```python
  # WORKER 0
  batch_sizes_0 = batch_sizes_for_worker(global_batch_size=global_batch_size,
                                         num_workers=2,
                                         num_replicas_per_worker=2,
                                         worker_index=0)
  print(batch_sizes_0)
  >> [2, 2, 2, 1]

  dataset_0 = tf.data.Dataset.from_tensor_slices(["file_a", "file_b"])
  dataset_0 = dataset_0.shard(num_shards, index=0)
  dataset_0 = dataset_0.batch(7)
  dataset_0 = dataset_0.apply(tf.data.experimental.rebatch(batch_sizes_0))
  for elem in dataset_0:
    print(elem)
  >> [[A0, A1], [A2, A3], [A4, A5], [A6]]

  # WORKER 1
  batch_sizes_1 = batch_sizes_for_worker(global_batch_size=global_batch_size,
                                         num_workers=2,
                                         num_replicas_per_worker=2,
                                         worker_index=1)
  print(batch_sizes_1)
  >> [2, 1, 2, 2]

  dataset_1 = tf.data.Dataset.from_tensor_slices(["file_a", "file_b"])
  dataset_1 = dataset_1.shard(num_shards, index=1)
  dataset_1 = dataset_1.batch(7)
  dataset_1 = dataset_1.apply(tf.data.experimental.rebatch(batch_sizes_1))
  for elem in dataset_1:
    print(elem)
  >> [[B0, B1], [B2], [B3, B4], [B5, B6]]
  ```

  The above example will produce the following elements:

  Step 1:
    Worker 0 Replica 0: [A0, A1]
    Worker 0 Replica 1: [A2, A3]
    Worker 1 Replica 0: [B0, B1]
    Worker 1 Replica 1: [B2]
  Total batch size = 7

  Step 2:
    Worker 0 Replica 0: [A4, A5]
    Worker 0 Replica 1: [A6]
    Worker 1 Replica 0: [B3, B4]
    Worker 1 Replica 1: [B5, B6]
  Total batch size = 7

  Args:
    global_batch_size: A `tf.int64` scalar, representing the global batch size.
    num_workers: An integer representing the number of workers the dataset will
      be distributed across.
    num_replicas_per_worker: An integer representing the number of replicas per
      worker. All workers are assumed to have the same number of replicas.
    worker_index: An integer index of the worker to be rebatched.

  Returns:
    A `tf.int64` vector, representing the batch sizes to rebatch the dataset
    into.
  """
    num_subbatches = num_workers * num_replicas_per_worker
    offset = worker_index * num_replicas_per_worker
    const_value = tensor_util.constant_value(global_batch_size)
    if const_value is not None:
        global_batch_size = const_value
    floor = global_batch_size // num_subbatches
    num_ceil = global_batch_size - num_subbatches * floor
    if const_value is not None:
        worker_0 = [floor + 1] * num_ceil + [floor] * (num_subbatches - num_ceil)
        return ops.convert_to_tensor(worker_0[offset:] + worker_0[:offset], dtype=dtypes.int64, name='batch_sizes')
    worker_0 = array_ops.ones(num_subbatches, dtype=dtypes.int64)
    worker_0 = floor * worker_0 + array_ops.concat([array_ops.ones(num_ceil, dtype=dtypes.int64), array_ops.zeros(num_subbatches - num_ceil, dtype=dtypes.int64)], axis=0)
    return array_ops.concat([worker_0[offset:], worker_0[:offset]], axis=0)