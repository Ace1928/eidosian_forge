import os
import shutil
import pytest
import tensorflow as tf
from tensorflow import keras
from autokeras import test_utils
from autokeras.utils import io_utils
def test_load_image_data_rgba():
    io_utils.image_dataset_from_directory(IMG_DATA_DIR, image_size=(180, 180), color_mode='rgba')