import os.path
import re
import sys
import shutil
from decimal import Decimal
from pyomo.common.dependencies import attempt_import
from pyomo.dataportal import TableData
from pyomo.dataportal.factory import DataManagerFactory
def del_source(self, name):
    """
        Remove an ODBC data source from the configuration. If
        any source specifications are based on this source, they
        will be removed as well.
        """
    if name in self.sources:
        if name in self.source_specs:
            del self.source_specs[name]
        del self.sources[name]