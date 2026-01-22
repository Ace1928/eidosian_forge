import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_remove_existing_attr(self):
    """
        Verify that an attribute value that evaluates to `None` removes an
        existing attribute of that name.
        """
    tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <elem class="foo" py:attrs="{\'class\': None}"/>\n        </doc>')
    self.assertEqual('<doc>\n          <elem/>\n        </doc>', tmpl.generate().render(encoding=None))