import os
import pyomo.common.unittest as unittest
def test_odbc_repr(self):
    config = ODBCConfig(data=self.simple_data)
    self.assertMultiLineEqual(config.odbc_repr(), self.simple_data)