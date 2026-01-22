import os
import json
import os.path
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.scripting.pyomo_main import main
from pyomo.scripting.util import cleanup
from pyomo.neos.kestrel import kestrelAMPL
import pyomo.neos
import pyomo.environ as pyo
from pyomo.common.fileutils import this_file_dir
def test_connection_failed(self):
    try:
        orig_host = pyomo.neos.kestrel.NEOS.host
        pyomo.neos.kestrel.NEOS.host = 'neos-bogus-server.org'
        with LoggingIntercept() as LOG:
            kestrel = kestrelAMPL()
        self.assertIsNone(kestrel.neos)
        self.assertRegex(LOG.getvalue(), 'NEOS is temporarily unavailable:\\n\\t\\(.+\\)')
    finally:
        pyomo.neos.kestrel.NEOS.host = orig_host