from io import BytesIO
from pprint import pformat
from ... import errors
from ...osutils import sha_string
from ..weave import Weave, WeaveFormatError, WeaveInvalidChecksum
from ..weavefile import read_weave, write_weave
from . import TestCase, TestCaseInTempDir
def test_compatible_parents(self):
    w1 = Weave('a')
    my_parents = {1, 2, 3}
    self.assertTrue(w1._compatible_parents(my_parents, {3}))
    self.assertTrue(w1._compatible_parents(my_parents, set(my_parents)))
    self.assertTrue(w1._compatible_parents(set(), set()))
    self.assertFalse(w1._compatible_parents(set(), {1}))
    self.assertFalse(w1._compatible_parents(my_parents, {1, 2, 3, 4}))
    self.assertFalse(w1._compatible_parents(my_parents, {4}))