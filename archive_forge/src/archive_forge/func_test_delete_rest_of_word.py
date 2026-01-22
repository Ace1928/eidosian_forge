import unittest
from bpython.curtsiesfrontend.manual_readline import (
def test_delete_rest_of_word(self):
    self.try_stages_kill(['z|s;df asdf d s;a;a', 'z|;df asdf d s;a;a', 'z| asdf d s;a;a', 'z| d s;a;a', 'z| s;a;a', 'z|;a;a', 'z|;a', 'z|', 'z|'], delete_rest_of_word)