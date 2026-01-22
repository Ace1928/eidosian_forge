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
def test_serializedAttributeWithTransparentTagWithRenderer(self) -> None:
    """
        Like L{test_serializedAttributeWithTransparentTag}, but when the
        attribute is rendered by a renderer on an element.
        """

    class WithRenderer(Element):

        def __init__(self, value: str, loader: Optional[ITemplateLoader]) -> None:
            self.value = value
            super().__init__(loader)

        @renderer
        def stuff(self, request: Optional[IRequest], tag: Tag) -> Flattenable:
            return self.value
    toss = []

    def insertRenderer(value: str) -> Flattenable:
        toss.append(value)
        return tags.transparent(render='stuff')

    def render(tag: Tag) -> Flattenable:
        return WithRenderer(toss.pop(), TagLoader(tag))
    self.checkAttributeSanitization(insertRenderer, render)