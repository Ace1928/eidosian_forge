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
def load_gzip_compressed_csv_data(data_file_name, *, data_module=DATA_MODULE, descr_file_name=None, descr_module=DESCR_MODULE, encoding='utf-8', **kwargs):
    """Loads gzip-compressed with `importlib.resources`.

    1) Open resource file with `importlib.resources.open_binary`
    2) Decompress file obj with `gzip.open`
    3) Load decompressed data with `np.loadtxt`

    Parameters
    ----------
    data_file_name : str
        Name of gzip-compressed csv file  (`'*.csv.gz'`) to be loaded from
        `data_module/data_file_name`. For example `'diabetes_data.csv.gz'`.

    data_module : str or module, default='sklearn.datasets.data'
        Module where data lives. The default is `'sklearn.datasets.data'`.

    descr_file_name : str, default=None
        Name of rst file to be loaded from `descr_module/descr_file_name`.
        For example `'wine_data.rst'`. See also :func:`load_descr`.
        If not None, also returns the corresponding description of
        the dataset.

    descr_module : str or module, default='sklearn.datasets.descr'
        Module where `descr_file_name` lives. See also :func:`load_descr`.
        The default  is `'sklearn.datasets.descr'`.

    encoding : str, default="utf-8"
        Name of the encoding that the gzip-decompressed file will be
        decoded with. The default is 'utf-8'.

    **kwargs : dict, optional
        Keyword arguments to be passed to `np.loadtxt`;
        e.g. delimiter=','.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features)
        A 2D array with each row representing one sample and each column
        representing the features and/or target of a given sample.

    descr : str, optional
        Description of the dataset (the content of `descr_file_name`).
        Only returned if `descr_file_name` is not None.
    """
    data_path = resources.files(data_module) / data_file_name
    with data_path.open('rb') as compressed_file:
        compressed_file = gzip.open(compressed_file, mode='rt', encoding=encoding)
        data = np.loadtxt(compressed_file, **kwargs)
    if descr_file_name is None:
        return data
    else:
        assert descr_module is not None
        descr = load_descr(descr_module=descr_module, descr_file_name=descr_file_name)
        return (data, descr)