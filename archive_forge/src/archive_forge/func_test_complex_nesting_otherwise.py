import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_complex_nesting_otherwise(self):
    """
        Verify more complex nesting using otherwise.
        """
    tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <div py:choose="1">\n            <div py:when="1" py:choose="2">\n              <span py:when="1">FAIL</span>\n              <span py:otherwise="">OK</span>\n            </div>\n          </div>\n        </doc>')
    self.assertEqual('<doc>\n          <div>\n            <div>\n              <span>OK</span>\n            </div>\n          </div>\n        </doc>', tmpl.generate().render(encoding=None))