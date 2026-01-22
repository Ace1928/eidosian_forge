import os
import traceback
from twisted.application import internet, service
from twisted.names import authority, dns, secondary, server
from twisted.python import usage
def opt_pyzone(self, filename):
    """Specify the filename of a Python syntax zone definition"""
    if not os.path.exists(filename):
        raise usage.UsageError(filename + ': No such file')
    self.zonefiles.append(filename)