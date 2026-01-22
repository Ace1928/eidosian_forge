import os
import shutil
import tempfile
from typing import TYPE_CHECKING, Optional
import tensorflow as tf
from tensorflow import keras
from ray.train._internal.framework_checkpoint import FrameworkCheckpoint
from ray.util.annotations import PublicAPI
@classmethod
def from_model(cls, model: keras.Model, *, preprocessor: Optional['Preprocessor']=None) -> 'TensorflowCheckpoint':
    """Create a :py:class:`~ray.train.Checkpoint` that stores a Keras model.

        The checkpoint created with this method needs to be paired with
        `model` when used.

        Args:
            model: The Keras model, whose weights are stored in the checkpoint.
            preprocessor: A fitted preprocessor to be applied before inference.

        Returns:
            A :py:class:`TensorflowCheckpoint` containing the specified model.

        Examples:

            .. testcode::

                from ray.train.tensorflow import TensorflowCheckpoint
                import tensorflow as tf

                model = tf.keras.applications.resnet.ResNet101()
                checkpoint = TensorflowCheckpoint.from_model(model)

            .. testoutput::
                :options: +MOCK
                :hide:

                ...  # Model may or may not be downloaded

        """
    tempdir = tempfile.mkdtemp()
    filename = 'model.keras'
    model.save(os.path.join(tempdir, filename))
    checkpoint = cls.from_directory(tempdir)
    if preprocessor:
        checkpoint.set_preprocessor(preprocessor)
    checkpoint.update_metadata({cls.MODEL_FILENAME_KEY: filename})
    return checkpoint