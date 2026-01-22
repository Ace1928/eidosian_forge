from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union
import click
import logging
import os
import time
import warnings
from ray.train._internal.storage import (
from ray.tune.experiment import Trial
from ray.tune.impl.out_of_band_serialize_dataset import out_of_band_serialize_dataset
def sync_up(self, force: bool=False, wait: bool=False) -> bool:
    syncer = self._storage.syncer
    if not syncer:
        return False
    exclude = _DRIVER_SYNC_EXCLUDE_PATTERNS
    experiment_local_path = self._storage.experiment_local_path
    experiment_fs_path = self._storage.experiment_fs_path
    if force:
        try:
            syncer.wait()
        except TimeoutError as e:
            logger.warning(f'The previous sync of the experiment directory to the cloud timed out with the error: {str(e)}\nSyncing will be retried. ' + _EXPERIMENT_SYNC_TIMEOUT_MESSAGE)
        except Exception as e:
            logger.warning(f'The previous sync of the experiment directory to the cloud failed with the error: {str(e)}\nSyncing will be retried.')
        synced = syncer.sync_up(local_dir=experiment_local_path, remote_dir=experiment_fs_path, exclude=exclude)
    else:
        synced = syncer.sync_up_if_needed(local_dir=experiment_local_path, remote_dir=experiment_fs_path, exclude=exclude)
    start_time = time.monotonic()
    if wait:
        try:
            syncer.wait()
        except Exception as e:
            raise RuntimeError(f'Uploading the experiment directory from the driver (local path: {experiment_local_path}) to the the cloud (remote path: {experiment_fs_path}) failed. Please check the error message above.') from e
    now = time.monotonic()
    sync_time_taken = now - start_time
    if sync_time_taken > self._slow_sync_threshold:
        try:
            import fsspec
        except Exception:
            fsspec = None
        fsspec_msg = ''
        if fsspec is None:
            fsspec_msg = 'If your data is small, try installing fsspec (`pip install fsspec`) for more efficient local file parsing. '
        logger.warning(f'Syncing the experiment checkpoint to cloud took a long time with {sync_time_taken:.2f} seconds. This can be due to a large number of trials, large logfiles, or throttling from the remote storage provider for too frequent syncs. {fsspec_msg}If your `CheckpointConfig.num_to_keep` is a low number, this can trigger frequent syncing, in which case you should increase it. ')
    if not synced:
        return False
    self._should_force_cloud_sync = False
    self._trial_num_checkpoints_since_last_sync.clear()
    if now - self._last_sync_time < self._excessive_sync_threshold:
        logger.warning(f'Experiment checkpoint syncing has been triggered multiple times in the last {self._excessive_sync_threshold} seconds. A sync will be triggered whenever a trial has checkpointed more than `num_to_keep` times since last sync or if {syncer.sync_period} seconds have passed since last sync. If you have set `num_to_keep` in your `CheckpointConfig`, consider increasing the checkpoint frequency or keeping more checkpoints. You can supress this warning by changing the `TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S` environment variable.')
    self._last_sync_time = now
    return True