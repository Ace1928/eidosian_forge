import sys
import types
import pytest
import pandas.util._test_decorators as td
@pytest.mark.parametrize('min_version,valid', [('0.0.0', True), ('99.99.99', False)])
def test_safe_import_versions(min_version, valid):
    result = td.safe_import('pandas', min_version=min_version)
    result = result if valid else not result
    assert result