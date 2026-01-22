import pickle
import unittest
from genshi import core
from genshi.core import Markup, Attrs, Namespace, QName, escape, unescape
from genshi.input import XML
from genshi.compat import StringIO, BytesIO, IS_PYTHON2
from genshi.tests.test_utils import doctest_suite
def test_render_utf8(self):
    xml = XML('<li>Über uns</li>')
    self.assertEqual(u'<li>Über uns</li>'.encode('utf-8'), xml.render(encoding='utf-8'))