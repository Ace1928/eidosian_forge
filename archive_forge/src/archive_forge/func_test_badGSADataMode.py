from __future__ import annotations
import datetime
from operator import attrgetter
from typing import Callable, Iterable, TypedDict
from zope.interface import implementer
from constantly import NamedConstant
from typing_extensions import Literal, Protocol
from twisted.positioning import base, ipositioning, nmea
from twisted.positioning.base import Angles
from twisted.positioning.test.receiver import MockPositioningReceiver
from twisted.trial.unittest import TestCase
def test_badGSADataMode(self) -> None:
    """
        GSA sentence data is not used when there is no GPS fix, but
        the data mode claims the data is "active". Some GPSes do
        this, unfortunately, and that means you shouldn't use the
        data.
        """
    sentenceData = {'type': 'GPGSA', 'altitude': '545.4', 'dataMode': nmea.GPGLLGPRMCFixQualities.ACTIVE, 'fixType': nmea.GPGSAFixTypes.GSA_NO_FIX}
    self._invalidFixTest(sentenceData)