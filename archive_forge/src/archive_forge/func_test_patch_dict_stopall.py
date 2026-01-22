import os
import sys
from collections import OrderedDict
import unittest
from unittest.test.testmock import support
from unittest.test.testmock.support import SomeClass, is_instance
from test.test_importlib.util import uncache
from unittest.mock import (
def test_patch_dict_stopall(self):
    dic1 = {}
    dic2 = {1: 'a'}
    dic3 = {1: 'A', 2: 'B'}
    origdic1 = dic1.copy()
    origdic2 = dic2.copy()
    origdic3 = dic3.copy()
    patch.dict(dic1, {1: 'I', 2: 'II'}).start()
    patch.dict(dic2, {2: 'b'}).start()

    @patch.dict(dic3)
    def patched():
        del dic3[1]
    patched()
    self.assertNotEqual(dic1, origdic1)
    self.assertNotEqual(dic2, origdic2)
    self.assertEqual(dic3, origdic3)
    patch.stopall()
    self.assertEqual(dic1, origdic1)
    self.assertEqual(dic2, origdic2)
    self.assertEqual(dic3, origdic3)