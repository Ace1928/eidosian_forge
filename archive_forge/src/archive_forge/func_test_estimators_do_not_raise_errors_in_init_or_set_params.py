import os
import pkgutil
import re
import sys
import warnings
from functools import partial
from inspect import isgenerator, signature
from itertools import chain, product
from pathlib import Path
import numpy as np
import pytest
import sklearn
from sklearn.cluster import (
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
from sklearn.experimental import (
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.model_selection import (
from sklearn.neighbors import (
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.utils import _IS_WASM, IS_PYPY, all_estimators
from sklearn.utils._tags import _DEFAULT_TAGS, _safe_tags
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
@pytest.mark.parametrize('Estimator', [est for name, est in all_estimators()])
def test_estimators_do_not_raise_errors_in_init_or_set_params(Estimator):
    """Check that init or set_param does not raise errors."""
    params = signature(Estimator).parameters
    smoke_test_values = [-1, 3.0, 'helloworld', np.array([1.0, 4.0]), [1], {}, []]
    for value in smoke_test_values:
        new_params = {key: value for key in params}
        est = Estimator(**new_params)
        est.set_params(**new_params)