from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
def test_read_param3(self):
    td = DataManagerFactory('xls')
    td.initialize(filename=currdir + 'Book1.xls', range='TheRange', index='X', param=['a'])
    try:
        td.open()
        td.read()
        td.close()
        self.assertEqual(td._info, ['param', ':', 'X', ':', 'a', ':=', 'A1', 2.0, 3.0, 4.0, 'A5', 6.0, 7.0, 8.0, 'A9', 10.0, 11.0, 12.0, 'A13', 14.0, 15.0, 16.0])
    except ApplicationError:
        pass