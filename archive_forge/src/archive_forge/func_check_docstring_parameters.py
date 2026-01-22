import atexit
import contextlib
import functools
import importlib
import inspect
import os
import os.path as op
import re
import shutil
import sys
import tempfile
import unittest
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from functools import wraps
from inspect import signature
from subprocess import STDOUT, CalledProcessError, TimeoutExpired, check_output
from unittest import TestCase
import joblib
import numpy as np
import scipy as sp
from numpy.testing import assert_allclose as np_assert_allclose
from numpy.testing import (
import sklearn
from sklearn.utils import (
from sklearn.utils._array_api import _check_array_api_dispatch
from sklearn.utils.fixes import VisibleDeprecationWarning, parse_version, sp_version
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (
def check_docstring_parameters(func, doc=None, ignore=None):
    """Helper to check docstring.

    Parameters
    ----------
    func : callable
        The function object to test.
    doc : str, default=None
        Docstring if it is passed manually to the test.
    ignore : list, default=None
        Parameters to ignore.

    Returns
    -------
    incorrect : list
        A list of string describing the incorrect results.
    """
    from numpydoc import docscrape
    incorrect = []
    ignore = [] if ignore is None else ignore
    func_name = _get_func_name(func)
    if not func_name.startswith('sklearn.') or func_name.startswith('sklearn.externals'):
        return incorrect
    if inspect.isdatadescriptor(func):
        return incorrect
    if func_name.split('.')[-1] in ('setup_module', 'teardown_module'):
        return incorrect
    if func_name.split('.')[2] == 'estimator_checks':
        return incorrect
    param_signature = list(filter(lambda x: x not in ignore, _get_args(func)))
    if len(param_signature) > 0 and param_signature[0] == 'self':
        param_signature.remove('self')
    if doc is None:
        records = []
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('error', UserWarning)
            try:
                doc = docscrape.FunctionDoc(func)
            except UserWarning as exp:
                if 'potentially wrong underline length' in str(exp):
                    message = str(exp).split('\n')[:3]
                    incorrect += [f'In function: {func_name}'] + message
                    return incorrect
                records.append(str(exp))
            except Exception as exp:
                incorrect += [func_name + ' parsing error: ' + str(exp)]
                return incorrect
        if len(records):
            raise RuntimeError('Error for %s:\n%s' % (func_name, records[0]))
    param_docs = []
    for name, type_definition, param_doc in doc['Parameters']:
        if not type_definition.strip():
            if ':' in name and name[:name.index(':')][-1:].strip():
                incorrect += [func_name + ' There was no space between the param name and colon (%r)' % name]
            elif name.rstrip().endswith(':'):
                incorrect += [func_name + ' Parameter %r has an empty type spec. Remove the colon' % name.lstrip()]
        if '*' not in name:
            param_docs.append(name.split(':')[0].strip('` '))
    if len(incorrect) > 0:
        return incorrect
    param_docs = list(filter(lambda x: x not in ignore, param_docs))
    message = []
    for i in range(min(len(param_docs), len(param_signature))):
        if param_signature[i] != param_docs[i]:
            message += ["There's a parameter name mismatch in function docstring w.r.t. function signature, at index %s diff: %r != %r" % (i, param_signature[i], param_docs[i])]
            break
    if len(param_signature) > len(param_docs):
        message += ['Parameters in function docstring have less items w.r.t. function signature, first missing item: %s' % param_signature[len(param_docs)]]
    elif len(param_signature) < len(param_docs):
        message += ['Parameters in function docstring have more items w.r.t. function signature, first extra item: %s' % param_docs[len(param_signature)]]
    if len(message) == 0:
        return []
    import difflib
    import pprint
    param_docs_formatted = pprint.pformat(param_docs).splitlines()
    param_signature_formatted = pprint.pformat(param_signature).splitlines()
    message += ['Full diff:']
    message.extend((line.strip() for line in difflib.ndiff(param_signature_formatted, param_docs_formatted)))
    incorrect.extend(message)
    incorrect = ['In function: ' + func_name] + incorrect
    return incorrect