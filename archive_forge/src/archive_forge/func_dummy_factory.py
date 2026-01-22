import pytest
from numpy.testing import assert_equal
from statsmodels.tools.decorators import (cache_readonly, deprecated_alias)
def dummy_factory(msg, remove_version, warning):

    class Dummy:
        y = deprecated_alias('y', 'x', remove_version=remove_version, msg=msg, warning=warning)

        def __init__(self, y):
            self.x = y
    return Dummy(1)