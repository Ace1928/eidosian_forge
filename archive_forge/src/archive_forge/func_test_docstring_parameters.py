import importlib
import inspect
import os
import warnings
from inspect import signature
from pkgutil import walk_packages
import numpy as np
import pytest
import sklearn
from sklearn.datasets import make_classification
from sklearn.experimental import (
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import IS_PYPY, all_estimators
from sklearn.utils._testing import (
from sklearn.utils.deprecation import _is_deprecated
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import parse_version, sp_version
@pytest.mark.filterwarnings('ignore::FutureWarning')
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
@pytest.mark.skipif(IS_PYPY, reason='test segfaults on PyPy')
def test_docstring_parameters():
    pytest.importorskip('numpydoc', reason='numpydoc is required to test the docstrings')
    from numpydoc import docscrape
    incorrect = []
    for name in PUBLIC_MODULES:
        if name.endswith('.conftest'):
            continue
        if name == 'sklearn.utils.fixes':
            continue
        with warnings.catch_warnings(record=True):
            module = importlib.import_module(name)
        classes = inspect.getmembers(module, inspect.isclass)
        classes = [cls for cls in classes if cls[1].__module__.startswith('sklearn')]
        for cname, cls in classes:
            this_incorrect = []
            if cname in _DOCSTRING_IGNORES or cname.startswith('_'):
                continue
            if inspect.isabstract(cls):
                continue
            with warnings.catch_warnings(record=True) as w:
                cdoc = docscrape.ClassDoc(cls)
            if len(w):
                raise RuntimeError('Error for __init__ of %s in %s:\n%s' % (cls, name, w[0]))
            if _is_deprecated(cls.__new__):
                continue
            this_incorrect += check_docstring_parameters(cls.__init__, cdoc)
            for method_name in cdoc.methods:
                method = getattr(cls, method_name)
                if _is_deprecated(method):
                    continue
                param_ignore = None
                if method_name in _METHODS_IGNORE_NONE_Y:
                    sig = signature(method)
                    if 'y' in sig.parameters and sig.parameters['y'].default is None:
                        param_ignore = ['y']
                result = check_docstring_parameters(method, ignore=param_ignore)
                this_incorrect += result
            incorrect += this_incorrect
        functions = inspect.getmembers(module, inspect.isfunction)
        functions = [fn for fn in functions if fn[1].__module__ == name]
        for fname, func in functions:
            if fname.startswith('_'):
                continue
            if fname == 'configuration' and name.endswith('setup'):
                continue
            name_ = _get_func_name(func)
            if not any((d in name_ for d in _DOCSTRING_IGNORES)) and (not _is_deprecated(func)):
                incorrect += check_docstring_parameters(func)
    msg = '\n'.join(incorrect)
    if len(incorrect) > 0:
        raise AssertionError('Docstring Error:\n' + msg)