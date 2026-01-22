import pytest
import sys
from pathlib import Path
import catalogue
@pytest.mark.skipif(sys.version_info < (3, 10) or sys.version_info >= (3, 12), reason='Test only supports python 3.10 and 3.11 importlib_metadata API')
def test_entry_points_newer():
    ep = catalogue.importlib_metadata.EntryPoint('bar', 'catalogue:check_exists', 'test_foo')
    catalogue.AVAILABLE_ENTRY_POINTS['test_foo'] = catalogue.importlib_metadata.EntryPoints([ep])
    _check_entry_points()