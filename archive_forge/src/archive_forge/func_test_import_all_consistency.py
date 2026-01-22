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
@pytest.mark.xfail(_IS_WASM, reason='importlib not supported for Pyodide packages')
@ignore_warnings
def test_import_all_consistency():
    sklearn_path = [os.path.dirname(sklearn.__file__)]
    pkgs = pkgutil.walk_packages(path=sklearn_path, prefix='sklearn.', onerror=lambda _: None)
    submods = [modname for _, modname, _ in pkgs]
    for modname in submods + ['sklearn']:
        if '.tests.' in modname:
            continue
        if 'sklearn._build_utils' in modname:
            continue
        if IS_PYPY and ('_svmlight_format_io' in modname or 'feature_extraction._hashing_fast' in modname):
            continue
        package = __import__(modname, fromlist='dummy')
        for name in getattr(package, '__all__', ()):
            assert hasattr(package, name), "Module '{0}' has no attribute '{1}'".format(modname, name)