import re
from pprint import PrettyPrinter
import numpy as np
from sklearn.utils._pprint import _EstimatorPrettyPrinter
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import config_context
def test_n_max_elements_to_show(print_changed_only_false):
    n_max_elements_to_show = 30
    pp = _EstimatorPrettyPrinter(compact=True, indent=1, indent_at_name=True, n_max_elements_to_show=n_max_elements_to_show)
    vocabulary = {i: i for i in range(n_max_elements_to_show)}
    vectorizer = CountVectorizer(vocabulary=vocabulary)
    expected = "\nCountVectorizer(analyzer='word', binary=False, decode_error='strict',\n                dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',\n                lowercase=True, max_df=1.0, max_features=None, min_df=1,\n                ngram_range=(1, 1), preprocessor=None, stop_words=None,\n                strip_accents=None, token_pattern='(?u)\\\\b\\\\w\\\\w+\\\\b',\n                tokenizer=None,\n                vocabulary={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,\n                            8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14,\n                            15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20,\n                            21: 21, 22: 22, 23: 23, 24: 24, 25: 25, 26: 26,\n                            27: 27, 28: 28, 29: 29})"
    expected = expected[1:]
    assert pp.pformat(vectorizer) == expected
    vocabulary = {i: i for i in range(n_max_elements_to_show + 1)}
    vectorizer = CountVectorizer(vocabulary=vocabulary)
    expected = "\nCountVectorizer(analyzer='word', binary=False, decode_error='strict',\n                dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',\n                lowercase=True, max_df=1.0, max_features=None, min_df=1,\n                ngram_range=(1, 1), preprocessor=None, stop_words=None,\n                strip_accents=None, token_pattern='(?u)\\\\b\\\\w\\\\w+\\\\b',\n                tokenizer=None,\n                vocabulary={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,\n                            8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14,\n                            15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20,\n                            21: 21, 22: 22, 23: 23, 24: 24, 25: 25, 26: 26,\n                            27: 27, 28: 28, 29: 29, ...})"
    expected = expected[1:]
    assert pp.pformat(vectorizer) == expected
    param_grid = {'C': list(range(n_max_elements_to_show))}
    gs = GridSearchCV(SVC(), param_grid)
    expected = "\nGridSearchCV(cv='warn', error_score='raise-deprecating',\n             estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,\n                           decision_function_shape='ovr', degree=3,\n                           gamma='auto_deprecated', kernel='rbf', max_iter=-1,\n                           probability=False, random_state=None, shrinking=True,\n                           tol=0.001, verbose=False),\n             iid='warn', n_jobs=None,\n             param_grid={'C': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,\n                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,\n                               27, 28, 29]},\n             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,\n             scoring=None, verbose=0)"
    expected = expected[1:]
    assert pp.pformat(gs) == expected
    param_grid = {'C': list(range(n_max_elements_to_show + 1))}
    gs = GridSearchCV(SVC(), param_grid)
    expected = "\nGridSearchCV(cv='warn', error_score='raise-deprecating',\n             estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,\n                           decision_function_shape='ovr', degree=3,\n                           gamma='auto_deprecated', kernel='rbf', max_iter=-1,\n                           probability=False, random_state=None, shrinking=True,\n                           tol=0.001, verbose=False),\n             iid='warn', n_jobs=None,\n             param_grid={'C': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,\n                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,\n                               27, 28, 29, ...]},\n             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,\n             scoring=None, verbose=0)"
    expected = expected[1:]
    assert pp.pformat(gs) == expected