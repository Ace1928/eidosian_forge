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
def test_translate_msgcontext_with_domain(self):
    tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:domain="bar" i18n:ctxt="foo">Foo, bar. <span>foo</span></p>\n          <p>Foo, bar.</p>\n        </html>')
    translations = DummyTranslations({('foo', 'Foo, bar.'): 'Fooo! Barrr!', 'Foo, bar.': 'Foo --- bar.'})
    translations.add_domain('bar', {('foo', 'foo'): 'BARRR', ('foo', 'Foo, bar.'): 'Bar, bar.'})
    translator = Translator(translations)
    translator.setup(tmpl)
    self.assertEqual('<html>\n          <p>Bar, bar. <span>BARRR</span></p>\n          <p>Foo --- bar.</p>\n        </html>', tmpl.generate().render())