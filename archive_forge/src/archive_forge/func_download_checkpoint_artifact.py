import logging
import os
import posixpath
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.utils.autologging_utils import (
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import LATEST_CHECKPOINT_ARTIFACT_TAG_KEY
def download_checkpoint_artifact(run_id=None, epoch=None, global_step=None, dst_path=None):
    from mlflow.client import MlflowClient
    from mlflow.utils.mlflow_tags import LATEST_CHECKPOINT_ARTIFACT_TAG_KEY
    client = MlflowClient()
    if run_id is None:
        run = mlflow.active_run()
        if run is None:
            raise MlflowException("There is no active run, please provide the 'run_id' argument for 'load_checkpoint' invocation.")
        run_id = run.info.run_id
    else:
        run = client.get_run(run_id)
    latest_checkpoint_artifact_path = run.data.tags.get(LATEST_CHECKPOINT_ARTIFACT_TAG_KEY)
    if latest_checkpoint_artifact_path is None:
        raise MlflowException('There is no logged checkpoint artifact in the current run.')
    checkpoint_filename = posixpath.basename(latest_checkpoint_artifact_path)
    if epoch is not None and global_step is not None:
        raise MlflowException("Only one of 'epoch' and 'global_step' can be set for 'load_checkpoint'.")
    elif global_step is not None:
        checkpoint_artifact_path = f'{_CHECKPOINT_DIR}/{_CHECKPOINT_GLOBAL_STEP_PREFIX}{global_step}/{checkpoint_filename}'
    elif epoch is not None:
        checkpoint_artifact_path = f'{_CHECKPOINT_DIR}/{_CHECKPOINT_EPOCH_PREFIX}{epoch}/{checkpoint_filename}'
    else:
        checkpoint_artifact_path = latest_checkpoint_artifact_path
    return client.download_artifacts(run_id, checkpoint_artifact_path, dst_path=dst_path)