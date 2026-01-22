import logging
import os
import posixpath
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.utils.autologging_utils import (
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import LATEST_CHECKPOINT_ARTIFACT_TAG_KEY
def check_and_save_checkpoint_if_needed(self, current_epoch, global_step, metric_dict):
    mlflow.set_tracking_uri(self.mlflow_tracking_uri)
    if self.save_best_only:
        if self.monitor not in metric_dict:
            _logger.warning("Checkpoint logging is skipped, because checkpoint 'save_best_only' config is True, it requires to compare the monitored metric value, but the provided monitored metric value is not available.")
            return
        new_monitor_value = metric_dict[self.monitor]
        if not self._is_new_checkpoint_better(new_monitor_value):
            self.last_monitor_value = new_monitor_value
            return
        self.last_monitor_value = new_monitor_value
    suffix = self.checkpoint_file_suffix
    if self.save_best_only:
        if self.save_weights_only:
            checkpoint_model_filename = f'{_LATEST_CHECKPOINT_PREFIX}{_CHECKPOINT_MODEL_FILENAME}{_WEIGHT_ONLY_CHECKPOINT_SUFFIX}{suffix}'
        else:
            checkpoint_model_filename = f'{_LATEST_CHECKPOINT_PREFIX}{_CHECKPOINT_MODEL_FILENAME}{suffix}'
        checkpoint_metrics_filename = f'{_LATEST_CHECKPOINT_PREFIX}{_CHECKPOINT_METRIC_FILENAME}'
        checkpoint_artifact_dir = _CHECKPOINT_DIR
    else:
        if self.save_freq == 'epoch':
            sub_dir_name = f'{_CHECKPOINT_EPOCH_PREFIX}{current_epoch}'
        else:
            sub_dir_name = f'{_CHECKPOINT_GLOBAL_STEP_PREFIX}{global_step}'
        if self.save_weights_only:
            checkpoint_model_filename = f'{_CHECKPOINT_MODEL_FILENAME}{_WEIGHT_ONLY_CHECKPOINT_SUFFIX}{suffix}'
        else:
            checkpoint_model_filename = f'{_CHECKPOINT_MODEL_FILENAME}{suffix}'
        checkpoint_metrics_filename = _CHECKPOINT_METRIC_FILENAME
        checkpoint_artifact_dir = f'{_CHECKPOINT_DIR}/{sub_dir_name}'
    mlflow.set_tag(LATEST_CHECKPOINT_ARTIFACT_TAG_KEY, f'{checkpoint_artifact_dir}/{checkpoint_model_filename}')
    mlflow.log_dict({**metric_dict, 'epoch': current_epoch, 'global_step': global_step}, f'{checkpoint_artifact_dir}/{checkpoint_metrics_filename}')
    with TempDir() as tmp_dir:
        tmp_model_save_path = os.path.join(tmp_dir.path(), checkpoint_model_filename)
        self.save_checkpoint(tmp_model_save_path)
        mlflow.log_artifact(tmp_model_save_path, checkpoint_artifact_dir)