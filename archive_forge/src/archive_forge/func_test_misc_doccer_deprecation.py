import pkgutil
import types
import importlib
import warnings
from importlib import import_module
import pytest
import scipy
def test_misc_doccer_deprecation():
    module = import_module('scipy.misc.doccer')
    correct_import = import_module('scipy._lib.doccer')
    for attr_name in module.__all__:
        attr = getattr(correct_import, attr_name, None)
        if attr is None:
            message = f'`scipy.misc.{attr_name}` is deprecated...'
        else:
            message = f'Please import `{attr_name}` from the `scipy._lib.doccer`...'
        with pytest.deprecated_call(match=message):
            getattr(module, attr_name)
    message = '`scipy.misc.doccer` is deprecated...'
    with pytest.raises(AttributeError, match=message):
        getattr(module, 'ekki')