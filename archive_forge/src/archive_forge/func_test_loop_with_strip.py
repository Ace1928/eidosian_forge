import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_loop_with_strip(self):
    """
        Verify that the combining the `py:if` directive with `py:strip` works
        correctly.
        """
    tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <b py:if="foo" py:strip="">${bar}</b>\n        </doc>')
    self.assertEqual('<doc>\n          Hello\n        </doc>', tmpl.generate(foo=True, bar='Hello').render(encoding=None))