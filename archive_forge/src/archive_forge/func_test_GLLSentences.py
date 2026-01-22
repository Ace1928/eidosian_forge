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
def test_GLLSentences(self) -> None:
    """
        GLL sentences fire C{positionReceived}.
        """
    sentences = [GPGLL_PARTIAL, GPGLL]
    self._receiverTest(sentences, ['positionReceived'])