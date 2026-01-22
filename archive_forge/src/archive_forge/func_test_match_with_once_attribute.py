import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_match_with_once_attribute(self):
    tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/">\n          <py:match path="body" once="true"><body>\n            <div id="wrap">\n              ${select("*")}\n            </div>\n          </body></py:match>\n          <body>\n            <p>Foo</p>\n          </body>\n          <body>\n            <p>Bar</p>\n          </body>\n        </html>')
    self.assertEqual('<html>\n          <body>\n            <div id="wrap">\n              <p>Foo</p>\n            </div>\n          </body>\n          <body>\n            <p>Bar</p>\n          </body>\n        </html>', tmpl.generate().render(encoding=None))