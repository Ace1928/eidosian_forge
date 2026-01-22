import pytest
from rpy2.rinterface_lib import openrlib
import rpy2.rinterface
def test_get_symbol_or_fallback():
    func = openrlib._get_symbol_or_fallback('thereisnosuchsymbol', lambda x: 'fallback')
    assert func(None) == 'fallback'