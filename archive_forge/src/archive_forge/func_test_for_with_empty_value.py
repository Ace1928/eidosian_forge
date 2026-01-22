import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_for_with_empty_value(self):
    """
        Verify an empty 'for' value is an error
        """
    try:
        MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n              <py:for each="">\n                empty\n              </py:for>\n            </doc>', filename='test.html').generate()
        self.fail('ExpectedTemplateSyntaxError')
    except TemplateSyntaxError as e:
        self.assertEqual('test.html', e.filename)
        if sys.version_info[:2] > (2, 4):
            self.assertEqual(2, e.lineno)