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
def test_serializeUnicode(self) -> None:
    """
        Test that unicode is encoded correctly in the appropriate places, and
        raises an error when it occurs in inappropriate place.
        """
    snowman = '☃'
    self.assertFlattensImmediately(snowman, b'\xe2\x98\x83')
    self.assertFlattensImmediately(tags.p(snowman), b'<p>\xe2\x98\x83</p>')
    self.assertFlattensImmediately(Comment(snowman), b'<!--\xe2\x98\x83-->')
    self.assertFlattensImmediately(CDATA(snowman), b'<![CDATA[\xe2\x98\x83]]>')
    self.assertFlatteningRaises(Tag(snowman), UnicodeEncodeError)
    self.assertFlatteningRaises(Tag('p', attributes={snowman: ''}), UnicodeEncodeError)