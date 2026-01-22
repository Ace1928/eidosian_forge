from itertools import product
import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import config_context, datasets
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVC
from sklearn.utils._array_api import yield_namespace_device_dtype_combinations
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import _NotAnArray
from sklearn.utils.fixes import (
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.multiclass import (
def test_type_of_target():
    for group, group_examples in EXAMPLES.items():
        for example in group_examples:
            assert type_of_target(example) == group, 'type_of_target(%r) should be %r, got %r' % (example, group, type_of_target(example))
    for example in NON_ARRAY_LIKE_EXAMPLES:
        msg_regex = 'Expected array-like \\(array or non-string sequence\\).*'
        with pytest.raises(ValueError, match=msg_regex):
            type_of_target(example)
    for example in MULTILABEL_SEQUENCES:
        msg = 'You appear to be using a legacy multi-label data representation. Sequence of sequences are no longer supported; use a binary array or sparse matrix instead.'
        with pytest.raises(ValueError, match=msg):
            type_of_target(example)