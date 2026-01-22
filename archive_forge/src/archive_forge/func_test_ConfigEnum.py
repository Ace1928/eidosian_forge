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
def test_ConfigEnum(self):
    out = StringIO()
    with LoggingIntercept(out):

        class TestEnum(ConfigEnum):
            ITEM_ONE = 1
            ITEM_TWO = 2
    self.assertIn('The ConfigEnum base class is deprecated', out.getvalue())
    self.assertEqual(TestEnum.from_enum_or_string(1), TestEnum.ITEM_ONE)
    self.assertEqual(TestEnum.from_enum_or_string(TestEnum.ITEM_TWO), TestEnum.ITEM_TWO)
    self.assertEqual(TestEnum.from_enum_or_string('ITEM_ONE'), TestEnum.ITEM_ONE)
    cfg = ConfigDict()
    cfg.declare('enum', ConfigValue(default=2, domain=TestEnum.from_enum_or_string))
    self.assertEqual(cfg.enum, TestEnum.ITEM_TWO)
    cfg.enum = 'ITEM_ONE'
    self.assertEqual(cfg.enum, TestEnum.ITEM_ONE)
    cfg.enum = TestEnum.ITEM_TWO
    self.assertEqual(cfg.enum, TestEnum.ITEM_TWO)
    cfg.enum = 1
    self.assertEqual(cfg.enum, TestEnum.ITEM_ONE)
    with self.assertRaisesRegex(ValueError, '.*3 is not a valid'):
        cfg.enum = 3
    with self.assertRaisesRegex(ValueError, '.*invalid value'):
        cfg.enum = 'ITEM_THREE'