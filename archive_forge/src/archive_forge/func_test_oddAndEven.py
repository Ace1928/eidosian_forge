import errno
import inspect
import mimetypes
import os
import re
import sys
import warnings
from io import BytesIO as StringIO
from unittest import skipIf
from zope.interface.verify import verifyObject
from twisted.internet import abstract, interfaces
from twisted.python import compat, log
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
from twisted.web import http, resource, script, static
from twisted.web._responses import FOUND
from twisted.web.server import UnsupportedMethod
from twisted.web.test._util import _render
from twisted.web.test.requesthelper import DummyRequest
def test_oddAndEven(self):
    """
        L{static.DirectoryLister} gives an alternate class for each odd and
        even rows in the table.
        """
    lister = static.DirectoryLister(None)
    elements = [{'href': '', 'text': '', 'size': '', 'type': '', 'encoding': ''} for i in range(5)]
    content = lister._buildTableContent(elements)
    self.assertEqual(len(content), 5)
    self.assertTrue(content[0].startswith('<tr class="odd">'))
    self.assertTrue(content[1].startswith('<tr class="even">'))
    self.assertTrue(content[2].startswith('<tr class="odd">'))
    self.assertTrue(content[3].startswith('<tr class="even">'))
    self.assertTrue(content[4].startswith('<tr class="odd">'))