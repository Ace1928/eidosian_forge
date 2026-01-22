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
@validate_params({'n_class': [Interval(Integral, 1, 10, closed='both')], 'return_X_y': ['boolean'], 'as_frame': ['boolean']}, prefer_skip_nested_validation=True)
def load_digits(*, n_class=10, return_X_y=False, as_frame=False):
    """Load and return the digits dataset (classification).

    Each datapoint is a 8x8 image of a digit.

    =================   ==============
    Classes                         10
    Samples per class             ~180
    Samples total                 1797
    Dimensionality                  64
    Features             integers 0-16
    =================   ==============

    This is a copy of the test set of the UCI ML hand-written digits datasets
    https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits

    Read more in the :ref:`User Guide <digits_dataset>`.

    Parameters
    ----------
    n_class : int, default=10
        The number of classes to return. Between 0 and 10.

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

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : {ndarray, dataframe} of shape (1797, 64)
            The flattened data matrix. If `as_frame=True`, `data` will be
            a pandas DataFrame.
        target: {ndarray, Series} of shape (1797,)
            The classification target. If `as_frame=True`, `target` will be
            a pandas Series.
        feature_names: list
            The names of the dataset columns.
        target_names: list
            The names of target classes.

            .. versionadded:: 0.20

        frame: DataFrame of shape (1797, 65)
            Only present when `as_frame=True`. DataFrame with `data` and
            `target`.

            .. versionadded:: 0.23
        images: {ndarray} of shape (1797, 8, 8)
            The raw image data.
        DESCR: str
            The full description of the dataset.

    (data, target) : tuple if ``return_X_y`` is True
        A tuple of two ndarrays by default. The first contains a 2D ndarray of
        shape (1797, 64) with each row representing one sample and each column
        representing the features. The second ndarray of shape (1797) contains
        the target samples.  If `as_frame=True`, both arrays are pandas objects,
        i.e. `X` a dataframe and `y` a series.

        .. versionadded:: 0.18

    Examples
    --------
    To load the data and visualize the images::

        >>> from sklearn.datasets import load_digits
        >>> digits = load_digits()
        >>> print(digits.data.shape)
        (1797, 64)
        >>> import matplotlib.pyplot as plt
        >>> plt.gray()
        >>> plt.matshow(digits.images[0])
        <...>
        >>> plt.show()
    """
    data, fdescr = load_gzip_compressed_csv_data(data_file_name='digits.csv.gz', descr_file_name='digits.rst', delimiter=',')
    target = data[:, -1].astype(int, copy=False)
    flat_data = data[:, :-1]
    images = flat_data.view()
    images.shape = (-1, 8, 8)
    if n_class < 10:
        idx = target < n_class
        flat_data, target = (flat_data[idx], target[idx])
        images = images[idx]
    feature_names = ['pixel_{}_{}'.format(row_idx, col_idx) for row_idx in range(8) for col_idx in range(8)]
    frame = None
    target_columns = ['target']
    if as_frame:
        frame, flat_data, target = _convert_data_dataframe('load_digits', flat_data, target, feature_names, target_columns)
    if return_X_y:
        return (flat_data, target)
    return Bunch(data=flat_data, target=target, frame=frame, feature_names=feature_names, target_names=np.arange(10), images=images, DESCR=fdescr)