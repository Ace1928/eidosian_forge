import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_select_all_attrs(self):
    tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <div py:match="elem" py:attrs="select(\'@*\')">\n            ${select(\'text()\')}\n          </div>\n          <elem id="joe">Hey Joe</elem>\n        </doc>')
    self.assertEqual('<doc>\n          <div id="joe">\n            Hey Joe\n          </div>\n        </doc>', tmpl.generate().render(encoding=None))