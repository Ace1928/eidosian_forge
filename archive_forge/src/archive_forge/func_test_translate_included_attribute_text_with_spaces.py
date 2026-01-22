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
def test_translate_included_attribute_text_with_spaces(self):
    tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/">\n          <span title=" Foo ">...</span>\n        </html>')
    translator = Translator(DummyTranslations({'Foo': 'Voh'}))
    translator.setup(tmpl)
    self.assertEqual('<html>\n          <span title="Voh">...</span>\n        </html>', tmpl.generate().render())