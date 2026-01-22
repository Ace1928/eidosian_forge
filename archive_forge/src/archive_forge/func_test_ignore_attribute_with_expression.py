from datetime import datetime
from gettext import NullTranslations
import unittest
import six
from genshi.core import Attrs
from genshi.template import MarkupTemplate, Context
from genshi.filters.i18n import Translator, extract
from genshi.input import HTML
from genshi.compat import IS_PYTHON2, StringIO
from genshi.tests.test_utils import doctest_suite
def test_ignore_attribute_with_expression(self):
    tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/">\n          <input type="submit" value="Reply" title="Reply to comment $num" />\n        </html>')
    translator = Translator()
    messages = list(translator.extract(tmpl.stream))
    self.assertEqual(0, len(messages))