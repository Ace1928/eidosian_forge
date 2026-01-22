import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_combined_with_loop(self):
    """
        Verify that the directive has access to the loop variables.
        """
    tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <elem py:for="item in items" py:attrs="item"/>\n        </doc>')
    items = [{'id': 1}, {'id': 2}]
    self.assertEqual('<doc>\n          <elem id="1"/><elem id="2"/>\n        </doc>', tmpl.generate(items=items).render(encoding=None))