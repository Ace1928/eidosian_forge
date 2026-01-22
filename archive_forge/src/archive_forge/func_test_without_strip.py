import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_without_strip(self):
    """
        Verify that a match template can produce the same kind of element that
        it matched without entering an infinite recursion.
        """
    tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <elem py:match="elem">\n            <div class="elem">${select(\'text()\')}</div>\n          </elem>\n          <elem>Hey Joe</elem>\n        </doc>')
    self.assertEqual('<doc>\n          <elem>\n            <div class="elem">Hey Joe</div>\n          </elem>\n        </doc>', tmpl.generate().render(encoding=None))