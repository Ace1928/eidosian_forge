import time
from tensorflow.python.eager import monitoring
from tensorflow.python.util import tf_contextlib
def get_metric_summary(metric_name):
    """Get summary for the specified metric."""
    metric = _METRICS_MAPPING[metric_name]
    result = metric.get_cell().value()
    if isinstance(metric, monitoring.Sampler):
        result = _get_metric_histogram(result)
    return result