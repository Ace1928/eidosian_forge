import unittest
from bpython import keys
def test_keymap_map(self):
    """Verify KeyMap.map being a dictionary with the correct
        length."""
    self.assertEqual(len(keys.urwid_key_dispatch.map), 64)