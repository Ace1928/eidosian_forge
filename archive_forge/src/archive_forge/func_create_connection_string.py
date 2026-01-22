import os.path
import re
import sys
import shutil
from decimal import Decimal
from pyomo.common.dependencies import attempt_import
from pyomo.dataportal import TableData
from pyomo.dataportal.factory import DataManagerFactory
def create_connection_string(self, ctype, connection, options):
    driver = self._get_driver(ctype)
    if driver:
        if ' ' in driver and (driver[0] != '{' or driver[-1] != '}'):
            return 'DRIVER={%s};Dbq=%s;' % (driver, connection)
        else:
            return 'DRIVER=%s;Dbq=%s;' % (driver, connection)
    return connection