import inspect
import os
import numpy as np
import pytest
import sklearn.datasets
def check_as_frame(bunch, dataset_func, expected_data_dtype=None, expected_target_dtype=None):
    pd = pytest.importorskip('pandas')
    frame_bunch = dataset_func(as_frame=True)
    assert hasattr(frame_bunch, 'frame')
    assert isinstance(frame_bunch.frame, pd.DataFrame)
    assert isinstance(frame_bunch.data, pd.DataFrame)
    assert frame_bunch.data.shape == bunch.data.shape
    if frame_bunch.target.ndim > 1:
        assert isinstance(frame_bunch.target, pd.DataFrame)
    else:
        assert isinstance(frame_bunch.target, pd.Series)
    assert frame_bunch.target.shape[0] == bunch.target.shape[0]
    if expected_data_dtype is not None:
        assert np.all(frame_bunch.data.dtypes == expected_data_dtype)
    if expected_target_dtype is not None:
        assert np.all(frame_bunch.target.dtypes == expected_target_dtype)
    frame_X, frame_y = dataset_func(as_frame=True, return_X_y=True)
    assert isinstance(frame_X, pd.DataFrame)
    if frame_y.ndim > 1:
        assert isinstance(frame_X, pd.DataFrame)
    else:
        assert isinstance(frame_y, pd.Series)