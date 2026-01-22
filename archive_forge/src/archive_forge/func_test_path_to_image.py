import os
import shutil
import pytest
import tensorflow as tf
from tensorflow import keras
from autokeras import test_utils
from autokeras.utils import io_utils
def test_path_to_image():
    img_dir = os.path.join(IMG_DATA_DIR, 'roses')
    assert isinstance(io_utils.path_to_image(os.path.join(img_dir, os.listdir(img_dir)[5]), num_channels=3, image_size=(180, 180), interpolation='bilinear'), tf.Tensor)