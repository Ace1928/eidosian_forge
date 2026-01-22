from typing import Dict, List, Tuple, cast
from zope.interface import implementer
from zope.interface.exceptions import BrokenMethodImplementation
from zope.interface.verify import verifyObject
from twisted.trial import unittest
from .._interfaces import ILogObserver, LogEvent
from .._logger import Logger
from .._observer import LogPublisher
@implementer(ILogObserver)
def o2(e: LogEvent) -> None:
    traces.setdefault(2, cast(Tuple[Tuple[Logger, ILogObserver]], tuple(e['log_trace'])))