import os
import string
import sys
from typing import Any, Dict, List, Optional, Union
import tensorflow as tf  # type: ignore
from tensorflow.keras import callbacks  # type: ignore
import wandb
from wandb.sdk.lib import telemetry
from wandb.sdk.lib.paths import StrPath
from ..keras import patch_tf_keras
@property
def is_old_tf_keras_version(self) -> Optional[bool]:
    if self._is_old_tf_keras_version is None:
        from wandb.util import parse_version
        try:
            if parse_version(tf.keras.__version__) < parse_version('2.6.0'):
                self._is_old_tf_keras_version = True
            else:
                self._is_old_tf_keras_version = False
        except AttributeError:
            self._is_old_tf_keras_version = False
    return self._is_old_tf_keras_version