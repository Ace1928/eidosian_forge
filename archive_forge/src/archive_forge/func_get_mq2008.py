import os
import zipfile
from dataclasses import dataclass
from typing import Any, Generator, List, NamedTuple, Optional, Tuple, Union
from urllib import request
import numpy as np
import pytest
from numpy import typing as npt
from numpy.random import Generator as RNG
from scipy import sparse
import xgboost
from xgboost.data import pandas_pyarrow_mapper
@memory.cache
def get_mq2008(dpath: str) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray, sparse.csr_matrix, np.ndarray, np.ndarray, sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Fetch the mq2008 dataset."""
    datasets = pytest.importorskip('sklearn.datasets')
    src = 'https://s3-us-west-2.amazonaws.com/xgboost-examples/MQ2008.zip'
    target = os.path.join(dpath, 'MQ2008.zip')
    if not os.path.exists(target):
        request.urlretrieve(url=src, filename=target)
    with zipfile.ZipFile(target, 'r') as f:
        f.extractall(path=dpath)
    x_train, y_train, qid_train, x_test, y_test, qid_test, x_valid, y_valid, qid_valid = datasets.load_svmlight_files((os.path.join(dpath, 'MQ2008/Fold1/train.txt'), os.path.join(dpath, 'MQ2008/Fold1/test.txt'), os.path.join(dpath, 'MQ2008/Fold1/vali.txt')), query_id=True, zero_based=False)
    return (x_train, y_train, qid_train, x_test, y_test, qid_test, x_valid, y_valid, qid_valid)