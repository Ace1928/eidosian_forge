import argparse
import enum
import os
import os.path
import pickle
import re
import sys
import types
import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.common.dependencies import yaml, yaml_available, yaml_load_args
from pyomo.common.config import (
from pyomo.common.log import LoggingIntercept
def test_IsInstance(self):
    c = ConfigDict()
    c.declare('val', ConfigValue(None, IsInstance(int)))
    c.val = 1
    self.assertEqual(c.val, 1)
    exc_str = "Expected an instance of 'int', but received value 2.4 of type 'float'"
    with self.assertRaisesRegex(ValueError, exc_str):
        c.val = 2.4

    class TestClass:

        def __repr__(self):
            return f'{TestClass.__name__}()'
    c.declare('val2', ConfigValue(None, IsInstance(TestClass)))
    testinst = TestClass()
    c.val2 = testinst
    self.assertEqual(c.val2, testinst)
    exc_str = "Expected an instance of 'TestClass', but received value 2.4 of type 'float'"
    with self.assertRaisesRegex(ValueError, exc_str):
        c.val2 = 2.4
    c.declare('val3', ConfigValue(None, IsInstance(int, TestClass, document_full_base_names=True)))
    self.assertRegex(c.get('val3').domain_name(), 'IsInstance\\(int, .*\\.TestClass\\)')
    c.val3 = 2
    self.assertEqual(c.val3, 2)
    exc_str = "Expected an instance of one of these types: 'int', '.*\\.TestClass', but received value 2.4 of type 'float'"
    with self.assertRaisesRegex(ValueError, exc_str):
        c.val3 = 2.4
    c.declare('val4', ConfigValue(None, IsInstance(int, TestClass, document_full_base_names=False)))
    self.assertEqual(c.get('val4').domain_name(), 'IsInstance(int, TestClass)')
    c.val4 = 2
    self.assertEqual(c.val4, 2)
    exc_str = "Expected an instance of one of these types: 'int', 'TestClass', but received value 2.4 of type 'float'"
    with self.assertRaisesRegex(ValueError, exc_str):
        c.val4 = 2.4