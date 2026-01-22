import pytest
from IPython.utils.importstring import import_item
def test_import_nested():
    """Test nested imports from the stdlib"""
    from os import path
    path2 = import_item('os.path')
    assert path is path2