import os
import sys
from collections import OrderedDict
import unittest
from unittest.test.testmock import support
from unittest.test.testmock.support import SomeClass, is_instance
from test.test_importlib.util import uncache
from unittest.mock import (
def test_patch_and_patch_dict_stopall(self):
    original_unlink = os.unlink
    original_chdir = os.chdir
    dic1 = {}
    dic2 = {1: 'A', 2: 'B'}
    origdic1 = dic1.copy()
    origdic2 = dic2.copy()
    patch('os.unlink', something).start()
    patch('os.chdir', something_else).start()
    patch.dict(dic1, {1: 'I', 2: 'II'}).start()
    patch.dict(dic2).start()
    del dic2[1]
    self.assertIsNot(os.unlink, original_unlink)
    self.assertIsNot(os.chdir, original_chdir)
    self.assertNotEqual(dic1, origdic1)
    self.assertNotEqual(dic2, origdic2)
    patch.stopall()
    self.assertIs(os.unlink, original_unlink)
    self.assertIs(os.chdir, original_chdir)
    self.assertEqual(dic1, origdic1)
    self.assertEqual(dic2, origdic2)