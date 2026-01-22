from mlflow.protos.facet_feature_statistics_pb2 import Histogram
def generate_equal_height_histogram(quantiles, num_buckets: int) -> Histogram:
    """
    Generates the equal height histogram from the input quantiles. The quantiles are assumed to
    be ordered and corresponding to equal distant percentiles.

    Args:
        quantiles: The quantiles that capture the frequency distribution.
        num_buckets: The number of buckets in the generated equal height histogram.

    Returns:
        An equal height histogram or None if inputs are invalid.

    """
    if len(quantiles) < 3 or (len(quantiles) - 1) % num_buckets != 0:
        return None
    histogram = Histogram()
    histogram.type = Histogram.HistogramType.QUANTILES
    step = (len(quantiles) - 1) // num_buckets
    for low_index in range(0, len(quantiles) - step, step):
        high_index = low_index + step
        histogram.buckets.append(Histogram.Bucket(low_value=quantiles[low_index], high_value=quantiles[high_index]))
    return histogram