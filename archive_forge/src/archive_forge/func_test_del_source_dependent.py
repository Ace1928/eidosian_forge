import os
import pyomo.common.unittest as unittest
def test_del_source_dependent(self):
    config = ODBCConfig()
    config.add_source('testdb', self.ACCESS_CONFIGSTR)
    config.add_source_spec('testdb', {'Database': 'testdb.mdb'})
    config.del_source('testdb')
    self.assertEqual({}, config.sources)
    self.assertEqual({}, config.source_specs)