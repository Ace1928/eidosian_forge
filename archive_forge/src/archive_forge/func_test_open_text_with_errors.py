import unittest
import importlib_resources as resources
from . import data01
from . import util
def test_open_text_with_errors(self):
    """
        Raises UnicodeError without the 'errors' argument.
        """
    target = resources.files(self.data) / 'utf-16.file'
    with target.open(encoding='utf-8', errors='strict') as fp:
        self.assertRaises(UnicodeError, fp.read)
    with target.open(encoding='utf-8', errors='ignore') as fp:
        result = fp.read()
    self.assertEqual(result, 'H\x00e\x00l\x00l\x00o\x00,\x00 \x00U\x00T\x00F\x00-\x001\x006\x00 \x00w\x00o\x00r\x00l\x00d\x00!\x00\n\x00')