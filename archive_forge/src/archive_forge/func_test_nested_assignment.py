import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_nested_assignment(self):
    """
        Verify that assignment to nested tuples works correctly.
        """
    tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <py:for each="idx, (k, v) in items">\n            <p>$idx: key=$k, value=$v</p>\n          </py:for>\n        </doc>')
    self.assertEqual('<doc>\n            <p>0: key=a, value=1</p>\n            <p>1: key=b, value=2</p>\n        </doc>', tmpl.generate(items=enumerate([('a', 1), ('b', 2)])).render(encoding=None))