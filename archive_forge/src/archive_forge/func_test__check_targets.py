import re
import warnings
from functools import partial
from itertools import chain, permutations, product
import numpy as np
import pytest
from scipy import linalg
from scipy.spatial.distance import hamming as sp_hamming
from scipy.stats import bernoulli
from sklearn import datasets, svm
from sklearn.datasets import make_multilabel_classification
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
from sklearn.metrics._classification import _check_targets
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelBinarizer, label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.extmath import _nanaverage
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.validation import check_random_state
def test__check_targets():
    IND = 'multilabel-indicator'
    MC = 'multiclass'
    BIN = 'binary'
    CNT = 'continuous'
    MMC = 'multiclass-multioutput'
    MCN = 'continuous-multioutput'
    EXAMPLES = [(IND, np.array([[0, 1, 1], [1, 0, 0], [0, 0, 1]])), (IND, np.array([[0, 1], [1, 0], [1, 1]])), (MC, [2, 3, 1]), (BIN, [0, 1, 1]), (CNT, [0.0, 1.5, 1.0]), (MC, np.array([[2], [3], [1]])), (BIN, np.array([[0], [1], [1]])), (CNT, np.array([[0.0], [1.5], [1.0]])), (MMC, np.array([[0, 2], [1, 3], [2, 3]])), (MCN, np.array([[0.5, 2.0], [1.1, 3.0], [2.0, 3.0]]))]
    EXPECTED = {(IND, IND): IND, (MC, MC): MC, (BIN, BIN): BIN, (MC, IND): None, (BIN, IND): None, (BIN, MC): MC, (CNT, CNT): None, (MMC, MMC): None, (MCN, MCN): None, (IND, CNT): None, (MC, CNT): None, (BIN, CNT): None, (MMC, CNT): None, (MCN, CNT): None, (IND, MMC): None, (MC, MMC): None, (BIN, MMC): None, (MCN, MMC): None, (IND, MCN): None, (MC, MCN): None, (BIN, MCN): None}
    for (type1, y1), (type2, y2) in product(EXAMPLES, repeat=2):
        try:
            expected = EXPECTED[type1, type2]
        except KeyError:
            expected = EXPECTED[type2, type1]
        if expected is None:
            with pytest.raises(ValueError):
                _check_targets(y1, y2)
            if type1 != type2:
                err_msg = "Classification metrics can't handle a mix of {0} and {1} targets".format(type1, type2)
                with pytest.raises(ValueError, match=err_msg):
                    _check_targets(y1, y2)
            elif type1 not in (BIN, MC, IND):
                err_msg = '{0} is not supported'.format(type1)
                with pytest.raises(ValueError, match=err_msg):
                    _check_targets(y1, y2)
        else:
            merged_type, y1out, y2out = _check_targets(y1, y2)
            assert merged_type == expected
            if merged_type.startswith('multilabel'):
                assert y1out.format == 'csr'
                assert y2out.format == 'csr'
            else:
                assert_array_equal(y1out, np.squeeze(y1))
                assert_array_equal(y2out, np.squeeze(y2))
            with pytest.raises(ValueError):
                _check_targets(y1[:-1], y2)
    y1 = [(1, 2), (0, 2, 3)]
    y2 = [(2,), (0, 2)]
    msg = 'You appear to be using a legacy multi-label data representation. Sequence of sequences are no longer supported; use a binary array or sparse matrix instead - the MultiLabelBinarizer transformer can convert to this format.'
    with pytest.raises(ValueError, match=msg):
        _check_targets(y1, y2)