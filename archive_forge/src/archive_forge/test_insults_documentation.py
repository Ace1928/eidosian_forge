import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest

        L{ServerProtocol.reportCursorPosition} writes a control
        sequence ending in L{CSFinalByte.DSR} with a parameter of 6
        (the Device Status Report returns the current active
        position.)
        