import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_when_without_test(self):
    """
        Verify that an `when` directive that doesn't have a `test` attribute
        is reported as an error.
        """
    tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <div py:choose="" py:strip="">\n            <py:when>foo</py:when>\n          </div>\n        </doc>')
    self.assertRaises(TemplateRuntimeError, str, tmpl.generate())