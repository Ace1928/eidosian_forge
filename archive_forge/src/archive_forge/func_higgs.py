import hashlib
import logging
import numpy as np
import os
import pandas as pd
import tarfile
import tempfile
import six
import shutil
from .core import PATH_TYPES, fspath
def higgs():
    """
    Download "higgs" [1] data set.

    Will return two pandas.DataFrame-s, first with train part and second with
    test part of the dataset. Object class will be located in the first
    column of dataset.

    [1]: https://archive.ics.uci.edu/ml/datasets/HIGGS
    """
    url = 'https://storage.mds.yandex.net/get-devtools-opensource/250854/higgs.tar.gz'
    md5 = 'ad59ba8328a9afa3837d7bf1a0e10e7b'
    dataset_name, train_file, test_file = ('higgs', 'train.tsv', 'test.tsv')
    train_path, test_path = _download_dataset(url, md5, dataset_name, train_file, test_file, cache=True)
    return (_load_numeric_only_dataset(train_path, 10500000, 29, sep='\t'), _load_numeric_only_dataset(test_path, 500000, 29, sep='\t'))