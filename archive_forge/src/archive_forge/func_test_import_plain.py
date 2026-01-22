import pytest
from IPython.utils.importstring import import_item
def test_import_plain():
    """Test simple imports"""
    import os
    os2 = import_item('os')
    assert os is os2