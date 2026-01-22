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
def flow_from_dataframe(self, dataframe, directory=None, x_col='filename', y_col='class', weight_col=None, target_size=(256, 256), color_mode='rgb', classes=None, class_mode='categorical', batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png', subset=None, interpolation='nearest', validate_filenames=True, **kwargs):
    if 'has_ext' in kwargs:
        warnings.warn('has_ext is deprecated, filenames in the dataframe have to match the exact filenames in disk.', DeprecationWarning)
    if 'sort' in kwargs:
        warnings.warn('sort is deprecated, batches will be created in thesame order than the filenames provided if shuffleis set to False.', DeprecationWarning)
    if class_mode == 'other':
        warnings.warn('`class_mode` "other" is deprecated, please use `class_mode` "raw".', DeprecationWarning)
        class_mode = 'raw'
    if 'drop_duplicates' in kwargs:
        warnings.warn('drop_duplicates is deprecated, you can drop duplicates by using the pandas.DataFrame.drop_duplicates method.', DeprecationWarning)
    return DataFrameIterator(dataframe, directory, self, x_col=x_col, y_col=y_col, weight_col=weight_col, target_size=target_size, color_mode=color_mode, classes=classes, class_mode=class_mode, data_format=self.data_format, batch_size=batch_size, shuffle=shuffle, seed=seed, save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format, subset=subset, interpolation=interpolation, validate_filenames=validate_filenames, dtype=self.dtype)