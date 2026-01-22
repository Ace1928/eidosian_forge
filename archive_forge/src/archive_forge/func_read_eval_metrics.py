import collections
import operator
import os
import tensorflow as tf
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
def read_eval_metrics(eval_dir):
    """Helper to read eval metrics from eval summary files.

  Args:
    eval_dir: Directory containing summary files with eval metrics.

  Returns:
    A `dict` with global steps mapping to `dict` of metric names and values.
  """
    eval_metrics_dict = collections.defaultdict(dict)
    for event in _summaries(eval_dir):
        if not event.HasField('summary'):
            continue
        metrics = {}
        for value in event.summary.value:
            if value.HasField('simple_value'):
                metrics[value.tag] = value.simple_value
        if metrics:
            eval_metrics_dict[event.step].update(metrics)
    return collections.OrderedDict(sorted(eval_metrics_dict.items(), key=lambda t: t[0]))