import os
import shutil
import pytest
import tensorflow as tf
from tensorflow import keras
from autokeras import test_utils
from autokeras.utils import io_utils
def test_load_image_data_raise_color_mode_error():
    with pytest.raises(ValueError) as info:
        io_utils.image_dataset_from_directory(IMG_DATA_DIR, image_size=(180, 180), color_mode='abcd')
    assert '`color_mode` must be one of' in str(info.value)