import re
import sys
import traceback
from collections import OrderedDict
from textwrap import dedent
from types import FunctionType
from typing import Callable, Dict, List, NoReturn, Optional, Tuple, cast
from xml.etree.ElementTree import XML
from zope.interface import implementer
from hamcrest import assert_that, equal_to
from twisted.internet.defer import (
from twisted.python.failure import Failure
from twisted.test.testutils import XMLAssertionMixin
from twisted.trial.unittest import SynchronousTestCase
from twisted.web._flatten import BUFFER_SIZE
from twisted.web.error import FlattenerError, UnfilledSlot, UnsupportedType
from twisted.web.iweb import IRenderable, IRequest, ITemplateLoader
from twisted.web.template import (
from twisted.web.test._util import FlattenTestCase
def test_chunksSeparatedByFiredDeferred(self) -> None:
    """
        When a fired L{Deferred} is encountered any buffered data is
        passed to the write function.  Then the L{Deferred}'s result is passed
        to another write along with following synchronous values.

        This exact buffering behavior should be considered an implementation
        detail and can be replaced by some other better behavior in the future
        if someone wants.
        """

    def sync_start(v: Flattenable) -> Tuple[Deferred[Flattenable], Callable[[], None]]:
        return (succeed(v), lambda: None)
    self._chunksSeparatedByAsyncTest(sync_start)