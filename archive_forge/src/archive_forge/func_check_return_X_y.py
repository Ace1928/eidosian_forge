import inspect
import os
import numpy as np
import pytest
import sklearn.datasets
def check_return_X_y(bunch, dataset_func):
    X_y_tuple = dataset_func(return_X_y=True)
    assert isinstance(X_y_tuple, tuple)
    assert X_y_tuple[0].shape == bunch.data.shape
    assert X_y_tuple[1].shape == bunch.target.shape