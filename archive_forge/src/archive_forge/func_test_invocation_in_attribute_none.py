import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_invocation_in_attribute_none(self):
    tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <py:def function="echo()">${None}</py:def>\n          <p class="${echo()}">bar</p>\n        </doc>')
    self.assertEqual('<doc>\n          <p>bar</p>\n        </doc>', tmpl.generate().render(encoding=None))