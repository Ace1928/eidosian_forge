import pytest
from geopandas._compat import import_optional_dependency
def test_import_optional_dependency_present():
    pandas = import_optional_dependency('pandas')
    assert pandas is not None
    import pandas as pd
    assert pandas == pd