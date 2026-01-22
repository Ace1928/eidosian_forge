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
def test_extract_i18n_choose_with_attributes(self):
    tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:choose="num; num" title="Things">\n            <i18n:singular>\n              There is <a href="$link" title="View thing">${num} thing</a>.\n            </i18n:singular>\n            <i18n:plural>\n              There are <a href="$link" title="View things">${num} things</a>.\n            </i18n:plural>\n          </p>\n        </html>')
    translator = Translator()
    translator.setup(tmpl)
    messages = list(translator.extract(tmpl.stream))
    self.assertEqual(4, len(messages))
    self.assertEqual((3, None, 'Things', []), messages[0])
    self.assertEqual((5, None, 'View thing', []), messages[1])
    self.assertEqual((8, None, 'View things', []), messages[2])
    self.assertEqual((3, 'ngettext', ('There is [1:%(num)s thing].', 'There are [1:%(num)s things].'), []), messages[3])