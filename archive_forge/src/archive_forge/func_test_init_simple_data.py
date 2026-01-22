import os
import pyomo.common.unittest as unittest
def test_init_simple_data(self):
    config = ODBCConfig(data=self.simple_data)
    self.assertEqual({'testdb': self.ACCESS_CONFIGSTR}, config.sources)
    self.assertEqual({'testdb': {'Database': 'testdb.mdb'}}, config.source_specs)
    self.assertEqual({}, config.odbc_info)