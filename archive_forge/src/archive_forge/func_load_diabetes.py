import csv
import gzip
import hashlib
import os
import shutil
from collections import namedtuple
from importlib import resources
from numbers import Integral
from os import environ, listdir, makedirs
from os.path import expanduser, isdir, join, splitext
from pathlib import Path
from urllib.request import urlretrieve
import numpy as np
from ..preprocessing import scale
from ..utils import Bunch, check_pandas_support, check_random_state
from ..utils._param_validation import Interval, StrOptions, validate_params
@validate_params({'return_X_y': ['boolean'], 'as_frame': ['boolean'], 'scaled': ['boolean']}, prefer_skip_nested_validation=True)
def load_diabetes(*, return_X_y=False, as_frame=False, scaled=True):
    """Load and return the diabetes dataset (regression).

    ==============   ==================
    Samples total    442
    Dimensionality   10
    Features         real, -.2 < x < .2
    Targets          integer 25 - 346
    ==============   ==================

    .. note::
       The meaning of each feature (i.e. `feature_names`) might be unclear
       (especially for `ltg`) as the documentation of the original dataset is
       not explicit. We provide information that seems correct in regard with
       the scientific literature in this field of research.

    Read more in the :ref:`User Guide <diabetes_dataset>`.

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

        .. versionadded:: 0.18

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.

        .. versionadded:: 0.23

    scaled : bool, default=True
        If True, the feature variables are mean centered and scaled by the
        standard deviation times the square root of `n_samples`.
        If False, raw data is returned for the feature variables.

        .. versionadded:: 1.1

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : {ndarray, dataframe} of shape (442, 10)
            The data matrix. If `as_frame=True`, `data` will be a pandas
            DataFrame.
        target: {ndarray, Series} of shape (442,)
            The regression target. If `as_frame=True`, `target` will be
            a pandas Series.
        feature_names: list
            The names of the dataset columns.
        frame: DataFrame of shape (442, 11)
            Only present when `as_frame=True`. DataFrame with `data` and
            `target`.

            .. versionadded:: 0.23
        DESCR: str
            The full description of the dataset.
        data_filename: str
            The path to the location of the data.
        target_filename: str
            The path to the location of the target.

    (data, target) : tuple if ``return_X_y`` is True
        Returns a tuple of two ndarray of shape (n_samples, n_features)
        A 2D array with each row representing one sample and each column
        representing the features and/or target of a given sample.

        .. versionadded:: 0.18

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> diabetes = load_diabetes()
    >>> diabetes.target[:3]
    array([151.,  75., 141.])
    >>> diabetes.data.shape
    (442, 10)
    """
    data_filename = 'diabetes_data_raw.csv.gz'
    target_filename = 'diabetes_target.csv.gz'
    data = load_gzip_compressed_csv_data(data_filename)
    target = load_gzip_compressed_csv_data(target_filename)
    if scaled:
        data = scale(data, copy=False)
        data /= data.shape[0] ** 0.5
    fdescr = load_descr('diabetes.rst')
    feature_names = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
    frame = None
    target_columns = ['target']
    if as_frame:
        frame, data, target = _convert_data_dataframe('load_diabetes', data, target, feature_names, target_columns)
    if return_X_y:
        return (data, target)
    return Bunch(data=data, target=target, frame=frame, DESCR=fdescr, feature_names=feature_names, data_filename=data_filename, target_filename=target_filename, data_module=DATA_MODULE)