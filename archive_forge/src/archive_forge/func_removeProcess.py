from typing import Dict, List, Optional
import attr
import incremental
from twisted.application import service
from twisted.internet import error, protocol, reactor as _reactor
from twisted.logger import Logger
from twisted.protocols import basic
from twisted.python import deprecate
def removeProcess(self, name):
    """
        Stop the named process and remove it from the list of monitored
        processes.

        @type name: C{str}
        @param name: A string that uniquely identifies the process.
        """
    self.stopProcess(name)
    del self._processes[name]