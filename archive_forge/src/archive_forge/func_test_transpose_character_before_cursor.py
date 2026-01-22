import unittest
from bpython.curtsiesfrontend.manual_readline import (
def test_transpose_character_before_cursor(self):
    self.try_stages(['as|df asdf', 'ads|f asdf', 'adfs| asdf', 'adf s|asdf', 'adf as|sdf'], transpose_character_before_cursor)