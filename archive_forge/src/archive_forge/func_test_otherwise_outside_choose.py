import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_otherwise_outside_choose(self):
    """
        Verify that an `otherwise` directive outside of a `choose` directive is
        reported as an error.
        """
    tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <div py:otherwise="" />\n        </doc>')
    self.assertRaises(TemplateRuntimeError, str, tmpl.generate())