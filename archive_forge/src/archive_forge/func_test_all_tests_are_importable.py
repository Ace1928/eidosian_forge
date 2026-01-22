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
@pytest.mark.skipif(sklearn._BUILT_WITH_MESON, reason='This test fails with Meson editable installs see https://github.com/mesonbuild/meson-python/issues/557 for more details')
def test_all_tests_are_importable():
    HAS_TESTS_EXCEPTIONS = re.compile('(?x)\n                                      \\.externals(\\.|$)|\n                                      \\.tests(\\.|$)|\n                                      \\._\n                                      ')
    resource_modules = {'sklearn.datasets.data', 'sklearn.datasets.descr', 'sklearn.datasets.images'}
    sklearn_path = [os.path.dirname(sklearn.__file__)]
    lookup = {name: ispkg for _, name, ispkg in pkgutil.walk_packages(sklearn_path, prefix='sklearn.')}
    missing_tests = [name for name, ispkg in lookup.items() if ispkg and name not in resource_modules and (not HAS_TESTS_EXCEPTIONS.search(name)) and (name + '.tests' not in lookup)]
    assert missing_tests == [], '{0} do not have `tests` subpackages. Perhaps they require __init__.py or an add_subpackage directive in the parent setup.py'.format(missing_tests)