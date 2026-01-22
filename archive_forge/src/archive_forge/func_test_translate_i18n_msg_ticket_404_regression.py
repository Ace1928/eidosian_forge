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
def test_translate_i18n_msg_ticket_404_regression(self):
    tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <h1 i18n:msg="name">text <a>$name</a></h1>\n        </html>')
    gettext = lambda s: u'head [1:%(name)s] tail'
    translator = Translator(gettext)
    translator.setup(tmpl)
    self.assertEqual('<html>\n          <h1>head <a>NAME</a> tail</h1>\n        </html>', tmpl.generate(name='NAME').render())