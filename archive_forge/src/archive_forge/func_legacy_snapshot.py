from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@deprecation.deprecated(None, 'Use `tf.data.Dataset.shapshot(...)` instead.')
def legacy_snapshot(path, compression=None, reader_path_prefix=None, writer_path_prefix=None, shard_size_bytes=None, pending_snapshot_expiry_seconds=None, num_reader_threads=None, reader_buffer_size=None, num_writer_threads=None, writer_buffer_size=None, shuffle_on_read=None, shuffle_seed=None, mode=None, snapshot_name=None):
    """Writes to/reads from a snapshot of a dataset.

  This function attempts to determine whether a valid snapshot exists at the
  `path`, and reads from the snapshot if so. If not, it will run the
  preprocessing pipeline as usual, and write out a snapshot of the data
  processed for future use.

  Args:
    path: A directory where we want to save our snapshots and/or read from a
      previously saved snapshot.
    compression: The type of compression to apply to the Dataset. Currently
      supports "GZIP" or None. Defaults to None (no compression).
    reader_path_prefix: A prefix to add to the path when reading from snapshots.
      Defaults to None.
    writer_path_prefix: A prefix to add to the path when writing to snapshots.
      Defaults to None.
    shard_size_bytes: The size of each shard to be written by the snapshot
      dataset op. Defaults to 10 GiB.
    pending_snapshot_expiry_seconds: How long to wait (in seconds) before the
      snapshot op considers a previously unfinished snapshot to be stale.
    num_reader_threads: Number of threads to parallelize reading from snapshot.
      Especially useful if compression is turned on since the decompression
      operation tends to be intensive. Defaults to 1. If > 1, then this might
      introduce non-determinism i.e. the order in which the elements are read
      from the snapshot are different from the order they're written.
    reader_buffer_size: Maximum number of elements we can prefetch reading from
      the snapshot. Defaults to 1. Increasing this might improve performance but
      will increase memory consumption.
    num_writer_threads: Number of threads to parallelize writing from snapshot.
      We'll open up `num_writer_threads` files and write to them in parallel.
      Especially useful if compression is turned on since the compression
      operation tends to be intensive. Defaults to 1. If > 1, then this might
      introduce non-determinism i.e. the order in which the elements are read
      from the upstream iterator are different from the order they're written.
    writer_buffer_size: Maximum number of pipeline elements to fill up the
      buffer before writing them out using `num_writer_threads`.
    shuffle_on_read: If this is True, then the order in which examples are
      produced when reading from a snapshot will be random. Defaults to False.
    shuffle_seed: Optional. If shuffle_seed is set, the random number generator
      used for shuffling (when shuffle_on_read is turned on) is seeded by the
      given seed. Otherwise, it is seeded by a random seed that differs for
      every run.
    mode: The mode at which snapshot should operate. Valid options are "auto",
      "read", "write", and "passthrough". The default mode is "auto", where the
      snapshot op will automatically determine what mode to operate in.
    snapshot_name: If set, use the supplied string as a named snapshot name
      instead of introspecting the data pipeline and automatically generating a
      unique identifier for the snapshot.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

    def _apply_fn(dataset):
        return _LegacySnapshotDataset(input_dataset=dataset, path=path, compression=compression, reader_path_prefix=reader_path_prefix, writer_path_prefix=writer_path_prefix, shard_size_bytes=shard_size_bytes, pending_snapshot_expiry_seconds=pending_snapshot_expiry_seconds, num_reader_threads=num_reader_threads, reader_buffer_size=reader_buffer_size, num_writer_threads=num_writer_threads, writer_buffer_size=writer_buffer_size, shuffle_on_read=shuffle_on_read, shuffle_seed=shuffle_seed, mode=mode, snapshot_name=snapshot_name)
    return _apply_fn