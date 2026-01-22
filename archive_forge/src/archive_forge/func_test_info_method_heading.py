import inspect
import sys
import pytest
import numpy as np
from numpy.core import arange
from numpy.testing import assert_, assert_equal, assert_raises_regex
from numpy.lib import deprecate, deprecate_with_doc
import numpy.lib.utils as utils
from io import StringIO
def test_info_method_heading():

    class NoPublicMethods:
        pass

    class WithPublicMethods:

        def first_method():
            pass

    def _has_method_heading(cls):
        out = StringIO()
        utils.info(cls, output=out)
        return 'Methods:' in out.getvalue()
    assert _has_method_heading(WithPublicMethods)
    assert not _has_method_heading(NoPublicMethods)