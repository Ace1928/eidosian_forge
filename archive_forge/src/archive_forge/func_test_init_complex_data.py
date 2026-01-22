import os
import pyomo.common.unittest as unittest
def test_init_complex_data(self):
    config = ODBCConfig(data=self.complex_data)
    self.assertEqual({'test1': self.ACCESS_CONFIGSTR, 'test2': self.EXCEL_CONFIGSTR}, config.sources)
    self.assertEqual({'test1': {'Database': 'test1.db', 'LogonID': 'Admin', 'pwd': 'secret_pass'}, 'test2': {'Database': 'test2.xls'}}, config.source_specs)
    self.assertEqual({'UNICODE': 'UTF-8'}, config.odbc_info)