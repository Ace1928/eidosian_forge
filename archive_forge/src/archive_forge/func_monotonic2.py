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
def monotonic2():
    """
    Dataset with monotonic constraints.
    Can be used for regression.
    The first column contains target values.
    Other columns contain contain numerical features, for which monotonic constraints must hold.

    For features in columns named MonotonicNeg*, if feature value decreases, then prediction
    value must not decrease. Thus, if there are two samples x1, x2 with all features being
    equal except for a monotonic negative feature MNeg, such that x1[MNeg] > x2[MNeg], then
    the following inequality must hold for predictions: f(x1) <= f(x2)
    For features in columns named MonotonicPos*, if feature value decreases, then prediction
    value must not increase. Thus, if there are two samples x1, x2 with all features being
    equal except for a monotonic positive feature MPos, such that x1[MPos] > x2[MPos],
    then the following inequality must hold for predictions: f(x1) >= f(x2)
    """
    url = 'https://storage.mds.yandex.net/get-devtools-opensource/250854/monotonic2.tar.gz'
    md5 = 'ce559e212cb72c156269f6f9a641baca'
    dataset_name, train_file, test_file = ('monotonic2', 'train.tsv', 'test.tsv')
    return _load_dataset_pd(url, md5, dataset_name, train_file, test_file, sep='\t')