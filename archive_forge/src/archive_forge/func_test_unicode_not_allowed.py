import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def test_unicode_not_allowed(self):
    lt_path_by_dirblock = self.get_lt_path_by_dirblock()
    self.assertRaises(TypeError, lt_path_by_dirblock, 'Uni', 'str')
    self.assertRaises(TypeError, lt_path_by_dirblock, 'str', 'Uni')
    self.assertRaises(TypeError, lt_path_by_dirblock, 'Uni', 'Uni')
    self.assertRaises(TypeError, lt_path_by_dirblock, 'x/Uni', 'x/str')
    self.assertRaises(TypeError, lt_path_by_dirblock, 'x/str', 'x/Uni')
    self.assertRaises(TypeError, lt_path_by_dirblock, 'x/Uni', 'x/Uni')