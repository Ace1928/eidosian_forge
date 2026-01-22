import collections
import operator
import os
import tensorflow as tf
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
@estimator_export('estimator.experimental.stop_if_lower_hook')
def stop_if_lower_hook(estimator, metric_name, threshold, eval_dir=None, min_steps=0, run_every_secs=60, run_every_steps=None):
    """Creates hook to stop if the given metric is lower than the threshold.

  Usage example:

  ```python
  estimator = ...
  # Hook to stop training if loss becomes lower than 100.
  hook = early_stopping.stop_if_lower_hook(estimator, "loss", 100)
  train_spec = tf.estimator.TrainSpec(..., hooks=[hook])
  tf.estimator.train_and_evaluate(estimator, train_spec, ...)
  ```

  Caveat: Current implementation supports early-stopping both training and
  evaluation in local mode. In distributed mode, training can be stopped but
  evaluation (where it's a separate job) will indefinitely wait for new model
  checkpoints to evaluate, so you will need other means to detect and stop it.
  Early-stopping evaluation in distributed mode requires changes in
  `train_and_evaluate` API and will be addressed in a future revision.

  Args:
    estimator: A `tf.estimator.Estimator` instance.
    metric_name: `str`, metric to track. "loss", "accuracy", etc.
    threshold: Numeric threshold for the given metric.
    eval_dir: If set, directory containing summary files with eval metrics. By
      default, `estimator.eval_dir()` will be used.
    min_steps: `int`, stop is never requested if global step is less than this
      value. Defaults to 0.
    run_every_secs: If specified, calls `should_stop_fn` at an interval of
      `run_every_secs` seconds. Defaults to 60 seconds. Either this or
      `run_every_steps` must be set.
    run_every_steps: If specified, calls `should_stop_fn` every
      `run_every_steps` steps. Either this or `run_every_secs` must be set.

  Returns:
    An early-stopping hook of type `SessionRunHook` that periodically checks
    if the given metric is lower than specified threshold and initiates
    early stopping if true.
  """
    return _stop_if_threshold_crossed_hook(estimator=estimator, metric_name=metric_name, threshold=threshold, higher_is_better=False, eval_dir=eval_dir, min_steps=min_steps, run_every_secs=run_every_secs, run_every_steps=run_every_steps)