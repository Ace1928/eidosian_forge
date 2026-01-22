import collections
import multiprocessing
import os
import threading
import warnings
import numpy as np
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.trainers.data_adapters.py_dataset_adapter import PyDataset
from keras.src.utils import image_utils
from keras.src.utils import io_utils
from keras.src.utils.module_utils import scipy
def random_transform(self, x, seed=None):
    """Applies a random transformation to an image.

        Args:
            x: 3D tensor, single image.
            seed: Random seed.

        Returns:
            A randomly transformed version of the input (same shape).
        """
    params = self.get_random_transform(x.shape, seed)
    return self.apply_transform(x, params)