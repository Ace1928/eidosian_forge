from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def new_model_fn(features, labels, mode, config):
    spec = estimator.model_fn(features, labels, mode, config)
    if mode != ModeKeys.EVAL:
        return spec
    new_metrics = _call_metric_fn(metric_fn, features, labels, spec.predictions, config)
    all_metrics = spec.eval_metric_ops or {}
    all_metrics.update(new_metrics)
    return spec._replace(eval_metric_ops=all_metrics)