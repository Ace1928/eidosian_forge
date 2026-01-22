from typing import Dict, List, Optional
import attr
import incremental
from twisted.application import service
from twisted.internet import error, protocol, reactor as _reactor
from twisted.logger import Logger
from twisted.protocols import basic
from twisted.python import deprecate
@deprecate.deprecatedProperty(incremental.Version('Twisted', 18, 7, 0))
def processes(self):
    """
        Processes as dict of tuples

        @return: Dict of process name to monitored processes as tuples
        """
    return {name: process.toTuple() for name, process in self._processes.items()}