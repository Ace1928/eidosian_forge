import unittest
from bpython import keys
def test_keymap_setitem(self):
    """Verify keys.KeyMap correctly setting items."""
    keys.urwid_key_dispatch['simon'] = 'awesome'
    self.assertEqual(keys.urwid_key_dispatch['simon'], 'awesome')