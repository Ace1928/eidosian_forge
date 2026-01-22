import os
import shutil
import tempfile
from typing import TYPE_CHECKING, Optional
import tensorflow as tf
from tensorflow import keras
from ray.train._internal.framework_checkpoint import FrameworkCheckpoint
from ray.util.annotations import PublicAPI
Retrieve the model stored in this checkpoint.

        Returns:
            The Tensorflow Keras model stored in the checkpoint.
        