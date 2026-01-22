import os
import pyomo.common.unittest as unittest
def test_add_spec_bad(self):
    config = ODBCConfig()
    with self.assertRaises(ODBCError):
        config.add_source_spec('testdb', {'Database': 'testdb.mdb'})