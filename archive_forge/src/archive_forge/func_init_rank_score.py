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
def init_rank_score(X: sparse.csr_matrix, y: npt.NDArray[np.int32], qid: npt.NDArray[np.int32], sample_rate: float=0.1) -> npt.NDArray[np.float32]:
    """We use XGBoost to generate the initial score instead of SVMRank for
    simplicity. Sample rate is set to 0.1 by default so that we can test with small
    datasets.

    """
    rng = np.random.default_rng(1994)
    n_samples = int(X.shape[0] * sample_rate)
    index = np.arange(0, X.shape[0], dtype=np.uint64)
    rng.shuffle(index)
    index = index[:n_samples]
    X_train = X[index]
    y_train = y[index]
    qid_train = qid[index]
    sorted_idx = np.argsort(qid_train)
    X_train = X_train[sorted_idx]
    y_train = y_train[sorted_idx]
    qid_train = qid_train[sorted_idx]
    ltr = xgboost.XGBRanker(objective='rank:ndcg', tree_method='hist')
    ltr.fit(X_train, y_train, qid=qid_train)
    scores = ltr.predict(X)
    return scores