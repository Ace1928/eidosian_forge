from functools import partial
from inspect import signature
from itertools import chain, permutations, product
import numpy as np
import pytest
from sklearn._config import config_context
from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import (
from sklearn.metrics._base import _average_binary_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from sklearn.utils._array_api import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _num_samples, check_random_state
@ignore_warnings
def check_sample_weight_invariance(name, metric, y1, y2):
    rng = np.random.RandomState(0)
    sample_weight = rng.randint(1, 10, size=len(y1))
    metric = partial(metric, k=1) if name == 'top_k_accuracy_score' else metric
    unweighted_score = metric(y1, y2, sample_weight=None)
    assert_allclose(unweighted_score, metric(y1, y2, sample_weight=np.ones(shape=len(y1))), err_msg='For %s sample_weight=None is not equivalent to sample_weight=ones' % name)
    weighted_score = metric(y1, y2, sample_weight=sample_weight)
    with pytest.raises(AssertionError):
        assert_allclose(unweighted_score, weighted_score)
        raise ValueError('Unweighted and weighted scores are unexpectedly almost equal (%s) and (%s) for %s' % (unweighted_score, weighted_score, name))
    weighted_score_list = metric(y1, y2, sample_weight=sample_weight.tolist())
    assert_allclose(weighted_score, weighted_score_list, err_msg='Weighted scores for array and list sample_weight input are not equal (%s != %s) for %s' % (weighted_score, weighted_score_list, name))
    repeat_weighted_score = metric(np.repeat(y1, sample_weight, axis=0), np.repeat(y2, sample_weight, axis=0), sample_weight=None)
    assert_allclose(weighted_score, repeat_weighted_score, err_msg='Weighting %s is not equal to repeating samples' % name)
    sample_weight_subset = sample_weight[1::2]
    sample_weight_zeroed = np.copy(sample_weight)
    sample_weight_zeroed[::2] = 0
    y1_subset = y1[1::2]
    y2_subset = y2[1::2]
    weighted_score_subset = metric(y1_subset, y2_subset, sample_weight=sample_weight_subset)
    weighted_score_zeroed = metric(y1, y2, sample_weight=sample_weight_zeroed)
    assert_allclose(weighted_score_subset, weighted_score_zeroed, err_msg='Zeroing weights does not give the same result as removing the corresponding samples (%s != %s) for %s' % (weighted_score_zeroed, weighted_score_subset, name))
    if not name.startswith('unnormalized'):
        for scaling in [2, 0.3]:
            assert_allclose(weighted_score, metric(y1, y2, sample_weight=sample_weight * scaling), err_msg='%s sample_weight is not invariant under scaling' % name)
    error_message = 'Found input variables with inconsistent numbers of samples: \\[{}, {}, {}\\]'.format(_num_samples(y1), _num_samples(y2), _num_samples(sample_weight) * 2)
    with pytest.raises(ValueError, match=error_message):
        metric(y1, y2, sample_weight=np.hstack([sample_weight, sample_weight]))