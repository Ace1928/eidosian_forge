import os
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.distribute import distributed_file_utils
from keras.src.utils import mode_keys
from keras.src.distribute.distributed_file_utils import (
def maybe_load_initial_counters_from_ckpt(self, steps_per_epoch, initial_epoch, mode):
    """Maybe load 1st epoch from checkpoint, considering worker recovery.

        When `_ckpt_saved_epoch` attribute exists and is not
        `CKPT_SAVED_EPOCH_UNUSED_VALUE`, this is under multi-worker training
        setting and indicates the worker is recovering from previous failure. In
        this case, infer `initial_epoch` from `self._ckpt_saved_epoch` to
        continue previous unfinished training from certain epoch.

        Args:
          steps_per_epoch: The number of steps per epoch value.
          initial_epoch: The original initial_epoch user passes in in `fit()`.
          mode: The mode for running `model.fit()`.

        Returns:
          If the training is recovering from previous failure under multi-worker
          training setting, return the (epoch, step) the training is supposed to
          continue at. Otherwise, return the `initial_epoch, initial_step` the
          user passes in.
        """
    initial_step = 0
    epoch = backend.eval(self._ckpt_saved_epoch)
    batch = backend.eval(self._ckpt_saved_batch)
    if mode == mode_keys.ModeKeys.TRAIN:
        if self._enable_save_before_preemption or isinstance(self._save_freq, int):
            if batch >= 0:
                if batch == steps_per_epoch - 1:
                    initial_epoch = epoch + 1
                    initial_step = 0
                else:
                    initial_epoch = epoch
                    initial_step = batch + 1
        elif epoch >= 0:
            initial_epoch = epoch + 1
    return (initial_epoch, initial_step)