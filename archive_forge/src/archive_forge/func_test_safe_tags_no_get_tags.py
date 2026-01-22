import pytest
from sklearn.base import BaseEstimator
from sklearn.utils._tags import (
@pytest.mark.parametrize('estimator, key, expected_results', [(NoTagsEstimator(), None, _DEFAULT_TAGS), (NoTagsEstimator(), 'allow_nan', _DEFAULT_TAGS['allow_nan']), (MoreTagsEstimator(), None, {**_DEFAULT_TAGS, **{'allow_nan': True}}), (MoreTagsEstimator(), 'allow_nan', True), (BaseEstimator(), None, _DEFAULT_TAGS), (BaseEstimator(), 'allow_nan', _DEFAULT_TAGS['allow_nan']), (BaseEstimator(), 'allow_nan', _DEFAULT_TAGS['allow_nan'])])
def test_safe_tags_no_get_tags(estimator, key, expected_results):
    assert _safe_tags(estimator, key=key) == expected_results