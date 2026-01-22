import pytest
import sys
from pathlib import Path
import catalogue
@pytest.mark.skipif(sys.version_info >= (3, 10), reason='Test does not support >=3.10 importlib_metadata API')
def test_entry_points_older():
    ep_string = '[options.entry_points]test_foo\n    bar = catalogue:check_exists'
    ep = catalogue.importlib_metadata.EntryPoint._from_text(ep_string)
    catalogue.AVAILABLE_ENTRY_POINTS['test_foo'] = ep
    _check_entry_points()