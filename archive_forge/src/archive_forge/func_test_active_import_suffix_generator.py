import os
import itertools
import logging
import pickle
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentMap
from pyomo.common.log import LoggingIntercept
from pyomo.core.base.suffix import (
from pyomo.environ import (
from io import StringIO
def test_active_import_suffix_generator(self):
    model = ConcreteModel()
    model.junk_IMPORT_int = Suffix(direction=Suffix.IMPORT, datatype=Suffix.INT)
    model.junk_IMPORT_float = Suffix(direction=Suffix.IMPORT, datatype=Suffix.FLOAT)
    model.junk_IMPORT_EXPORT_float = Suffix(direction=Suffix.IMPORT_EXPORT, datatype=Suffix.FLOAT)
    model.junk_EXPORT = Suffix(direction=Suffix.EXPORT, datatype=None)
    model.junk_LOCAL = Suffix(direction=Suffix.LOCAL, datatype=None)
    suffixes = dict(active_import_suffix_generator(model))
    self.assertTrue('junk_IMPORT_int' in suffixes)
    self.assertTrue('junk_IMPORT_float' in suffixes)
    self.assertTrue('junk_IMPORT_EXPORT_float' in suffixes)
    self.assertTrue('junk_EXPORT' not in suffixes)
    self.assertTrue('junk_LOCAL' not in suffixes)
    model.junk_IMPORT_float.deactivate()
    suffixes = dict(active_import_suffix_generator(model))
    self.assertTrue('junk_IMPORT_int' in suffixes)
    self.assertTrue('junk_IMPORT_float' not in suffixes)
    self.assertTrue('junk_IMPORT_EXPORT_float' in suffixes)
    self.assertTrue('junk_EXPORT' not in suffixes)
    self.assertTrue('junk_LOCAL' not in suffixes)
    model.junk_IMPORT_float.activate()
    suffixes = dict(active_import_suffix_generator(model, datatype=Suffix.FLOAT))
    self.assertTrue('junk_IMPORT_int' not in suffixes)
    self.assertTrue('junk_IMPORT_float' in suffixes)
    self.assertTrue('junk_IMPORT_EXPORT_float' in suffixes)
    self.assertTrue('junk_EXPORT' not in suffixes)
    self.assertTrue('junk_LOCAL' not in suffixes)
    model.junk_IMPORT_float.deactivate()
    suffixes = dict(active_import_suffix_generator(model, datatype=Suffix.FLOAT))
    self.assertTrue('junk_IMPORT_int' not in suffixes)
    self.assertTrue('junk_IMPORT_float' not in suffixes)
    self.assertTrue('junk_IMPORT_EXPORT_float' in suffixes)
    self.assertTrue('junk_EXPORT' not in suffixes)
    self.assertTrue('junk_LOCAL' not in suffixes)