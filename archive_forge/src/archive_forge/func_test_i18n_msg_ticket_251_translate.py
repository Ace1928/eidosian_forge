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
def test_i18n_msg_ticket_251_translate(self):
    tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg=""><tt><b>Translation[&nbsp;0&nbsp;]</b>: <em>One coin</em></tt></p>\n        </html>')
    translations = DummyTranslations({u'[1:[2:Translation\\[\xa00\xa0\\]]: [3:One coin]]': u'[1:[2:Trandução\\[\xa00\xa0\\]]: [3:Uma moeda]]'})
    translator = Translator(translations)
    translator.setup(tmpl)
    self.assertEqual(u'<html>\n          <p><tt><b>Trandução[\xa00\xa0]</b>: <em>Uma moeda</em></tt></p>\n        </html>'.encode('utf-8'), tmpl.generate().render(encoding='utf-8'))