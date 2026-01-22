import unittest
from bpython import keys
def test_keymap_delitem(self):
    """Verify keys.KeyMap correctly removing items."""
    keys.urwid_key_dispatch['simon'] = 'awesome'
    del keys.urwid_key_dispatch['simon']
    if 'simon' in keys.urwid_key_dispatch.map:
        raise Exception('Key still exists in dictionary')