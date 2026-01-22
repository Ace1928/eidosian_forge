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
def test_check_classification_targets():
    for y_type in EXAMPLES.keys():
        if y_type in ['unknown', 'continuous', 'continuous-multioutput']:
            for example in EXAMPLES[y_type]:
                msg = 'Unknown label type: '
                with pytest.raises(ValueError, match=msg):
                    check_classification_targets(example)
        else:
            for example in EXAMPLES[y_type]:
                check_classification_targets(example)