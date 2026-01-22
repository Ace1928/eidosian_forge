import numpy as np
from tensorboard.compat import tf2 as tf
from tensorboard.compat.proto import summary_pb2
from tensorboard.plugins.histogram import metadata
from tensorboard.util import lazy_tensor_creator
from tensorboard.util import tensor_util
def histogram_pb(tag, data, buckets=None, description=None):
    """Create a histogram summary protobuf.

    Arguments:
      tag: String tag for the summary.
      data: A `np.array` or array-like form of any shape. Must have type
        castable to `float`.
      buckets: Optional positive `int`. The output shape will always be
        [buckets, 3]. If there is no data, then an all-zero array of shape
        [buckets, 3] will be returned. If there is data but all points have
        the same value, then all buckets' left and right endpoints are the
        same and only the last bucket has nonzero count. Defaults to 30 if
        not specified.
      description: Optional long-form description for this summary, as a
        `str`. Markdown is supported. Defaults to empty.

    Returns:
      A `summary_pb2.Summary` protobuf object.
    """
    bucket_count = DEFAULT_BUCKET_COUNT if buckets is None else buckets
    data = np.array(data).flatten().astype(float)
    if bucket_count == 0 or data.size == 0:
        histogram_buckets = np.zeros((bucket_count, 3))
    else:
        min_ = np.min(data)
        max_ = np.max(data)
        range_ = max_ - min_
        if range_ == 0:
            left_edges = right_edges = np.array([min_] * bucket_count)
            bucket_counts = np.array([0] * (bucket_count - 1) + [data.size])
            histogram_buckets = np.array([left_edges, right_edges, bucket_counts]).transpose()
        else:
            bucket_width = range_ / bucket_count
            offsets = data - min_
            bucket_indices = np.floor(offsets / bucket_width).astype(int)
            clamped_indices = np.minimum(bucket_indices, bucket_count - 1)
            one_hots = np.array([clamped_indices]).transpose() == np.arange(0, bucket_count)
            assert one_hots.shape == (data.size, bucket_count), (one_hots.shape, (data.size, bucket_count))
            bucket_counts = np.sum(one_hots, axis=0)
            edges = np.linspace(min_, max_, bucket_count + 1)
            left_edges = edges[:-1]
            right_edges = edges[1:]
            histogram_buckets = np.array([left_edges, right_edges, bucket_counts]).transpose()
    tensor = tensor_util.make_tensor_proto(histogram_buckets, dtype=np.float64)
    summary_metadata = metadata.create_summary_metadata(display_name=None, description=description)
    summary = summary_pb2.Summary()
    summary.value.add(tag=tag, metadata=summary_metadata, tensor=tensor)
    return summary