import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_match_with_recursive_attribute(self):
    tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <py:match path="elem" recursive="false"><elem>\n            <div class="elem">\n              ${select(\'*\')}\n            </div>\n          </elem></py:match>\n          <elem>\n            <subelem>\n              <elem/>\n            </subelem>\n          </elem>\n        </doc>')
    self.assertEqual('<doc>\n          <elem>\n            <div class="elem">\n              <subelem>\n              <elem/>\n            </subelem>\n            </div>\n          </elem>\n        </doc>', tmpl.generate().render(encoding=None))