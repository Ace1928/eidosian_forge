from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import json
import os
import time
import six
import tensorflow as tf
from tensorflow.python.distribute import estimator_training as distribute_coordinator_training
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import server_lib
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator import exporter as exporter_lib
from tensorflow_estimator.python.estimator import run_config as run_config_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
def evaluate_and_export(self):
    """Evaluate and (maybe) export the current model.

      Returns:
        A tuple of `EvalResult` instance and the export results.

      Raises:
        RuntimeError: for any unexpected internal error.
        TypeError: if evaluation result has wrong type.
      """
    latest_ckpt_path = self._estimator.latest_checkpoint()
    if not latest_ckpt_path:
        self._log_err_msg('Estimator is not trained yet. Will start an evaluation when a checkpoint is ready.')
        return (_EvalResult(status=_EvalStatus.MISSING_CHECKPOINT), [])
    if latest_ckpt_path == self._previous_ckpt_path:
        self._log_err_msg('No new checkpoint ready for evaluation. Skip the current evaluation pass as evaluation results are expected to be same for the same checkpoint.')
        return (_EvalResult(status=_EvalStatus.NO_NEW_CHECKPOINT), [])
    metrics = self._estimator.evaluate(input_fn=self._eval_spec.input_fn, steps=self._eval_spec.steps, name=self._eval_spec.name, checkpoint_path=latest_ckpt_path, hooks=self._eval_spec.hooks)
    eval_result = _EvalResult(status=_EvalStatus.EVALUATED, metrics=metrics, checkpoint_path=latest_ckpt_path)
    is_the_final_export = eval_result.metrics[tf.compat.v1.GraphKeys.GLOBAL_STEP] >= self._max_training_steps if self._max_training_steps else False
    export_results = self._export_eval_result(eval_result, is_the_final_export)
    if is_the_final_export:
        tf.compat.v1.logging.debug('Calling exporter with the `is_the_final_export=True`.')
        self._is_final_export_triggered = True
    self._last_warning_time = 0
    self._previous_ckpt_path = latest_ckpt_path
    return (eval_result, export_results)