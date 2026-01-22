import sys
import warnings
import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared.utils import (
@pytest.mark.skipif(not have_numpydoc, reason='requires numpydoc')
def test_docstring_replaced_param(self):
    assert _func_replace_params.__name__ == '_func_replace_params'
    if sys.flags.optimize < 2:
        assert _func_replace_params.__doc__ == 'Expected docstring.\n\n\n    Parameters\n    ----------\n    arg0 : int\n        First unchanged parameter.\n    new0 : int, optional\n        First new parameter.\n\n        .. versionadded:: 0.10\n    new1 : int, optional\n        Second new parameter.\n\n        .. versionadded:: 0.10\n    arg1 : int, optional\n        Second unchanged parameter.\n\n    Other Parameters\n    ----------------\n    old0 : DEPRECATED\n        Deprecated in favor of `new1`.\n\n        .. deprecated:: 0.10\n    old1 : DEPRECATED\n        Deprecated in favor of `new0`.\n\n        .. deprecated:: 0.10\n'