from mlflow.protos.facet_feature_statistics_pb2 import Histogram

    Generates a histogram bucket given the bucket range, the quantiles and the total frequency.

    Args:
        bucket_left: Bucket left boundary.
        bucket_right: Bucket right boundary.
        quantiles: The quantiles that capture the frequency distribution.
        total_freq: The total frequency (=count of rows).

    Returns:
        The histogram bucket corresponding to the inputs.
    