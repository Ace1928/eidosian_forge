import numpy as np
from tensorboard.plugins.pr_curve import metadata
def raw_data_pb(name, true_positive_counts, false_positive_counts, true_negative_counts, false_negative_counts, precision, recall, num_thresholds=None, display_name=None, description=None):
    """Create a PR curves summary protobuf from raw data values.

    Args:
      name: A tag attached to the summary. Used by TensorBoard for organization.
      true_positive_counts: A rank-1 numpy array of true positive counts. Must
          contain `num_thresholds` elements and be castable to float32.
      false_positive_counts: A rank-1 numpy array of false positive counts. Must
          contain `num_thresholds` elements and be castable to float32.
      true_negative_counts: A rank-1 numpy array of true negative counts. Must
          contain `num_thresholds` elements and be castable to float32.
      false_negative_counts: A rank-1 numpy array of false negative counts. Must
          contain `num_thresholds` elements and be castable to float32.
      precision: A rank-1 numpy array of precision values. Must contain
          `num_thresholds` elements and be castable to float32.
      recall: A rank-1 numpy array of recall values. Must contain `num_thresholds`
          elements and be castable to float32.
      num_thresholds: Number of thresholds, evenly distributed in `[0, 1]`, to
          compute PR metrics for. Should be an int `>= 2`.
      display_name: Optional name for this summary in TensorBoard, as a `str`.
          Defaults to `name`.
      description: Optional long-form description for this summary, as a `str`.
          Markdown is supported. Defaults to empty.

    Returns:
      A summary operation for use in a TensorFlow graph. See docs for the `op`
      method for details on the float32 tensor produced by this summary.
    """
    import tensorflow.compat.v1 as tf
    if display_name is None:
        display_name = name
    summary_metadata = metadata.create_summary_metadata(display_name=display_name if display_name is not None else name, description=description or '', num_thresholds=num_thresholds)
    tf_summary_metadata = tf.SummaryMetadata.FromString(summary_metadata.SerializeToString())
    summary = tf.Summary()
    data = np.stack((true_positive_counts, false_positive_counts, true_negative_counts, false_negative_counts, precision, recall))
    tensor = tf.make_tensor_proto(np.float32(data), dtype=tf.float32)
    summary.value.add(tag='%s/pr_curves' % name, metadata=tf_summary_metadata, tensor=tensor)
    return summary