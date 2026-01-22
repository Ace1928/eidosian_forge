from __future__ import annotations
import pandas._testing as tm
from pandas.api import types
from pandas.tests.api.test_api import Base
def test_deprecated_from_api_types(self):
    for t in self.deprecated:
        with tm.assert_produces_warning(FutureWarning):
            getattr(types, t)(1)