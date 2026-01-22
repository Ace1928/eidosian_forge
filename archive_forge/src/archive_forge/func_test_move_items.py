import operator
import sys
import types
import unittest
import abc
import pytest
import six
@pytest.mark.parametrize('item_name', [item.name for item in six._moved_attributes])
def test_move_items(item_name):
    """Ensure that everything loads correctly."""
    try:
        item = getattr(six.moves, item_name)
        if isinstance(item, types.ModuleType):
            __import__('six.moves.' + item_name)
    except ImportError:
        if item_name == 'winreg' and (not sys.platform.startswith('win')):
            pytest.skip('Windows only module')
        if item_name.startswith('tkinter'):
            if not have_tkinter:
                pytest.skip('requires tkinter')
        if item_name.startswith('dbm_gnu') and (not have_gdbm):
            pytest.skip('requires gdbm')
        raise
    assert item_name in dir(six.moves)