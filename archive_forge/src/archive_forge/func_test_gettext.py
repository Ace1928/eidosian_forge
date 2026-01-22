import datetime
import os
import shutil
import tempfile
import unittest
import tornado.locale
from tornado.escape import utf8, to_unicode
from tornado.util import unicode_type
def test_gettext(self):
    tornado.locale.load_gettext_translations(os.path.join(os.path.dirname(__file__), 'gettext_translations'), 'tornado_test')
    locale = tornado.locale.get('fr_FR')
    self.assertTrue(isinstance(locale, tornado.locale.GettextLocale))
    self.assertEqual(locale.translate('school'), 'école')
    self.assertEqual(locale.pgettext('law', 'right'), 'le droit')
    self.assertEqual(locale.pgettext('good', 'right'), 'le bien')
    self.assertEqual(locale.pgettext('organization', 'club', 'clubs', 1), 'le club')
    self.assertEqual(locale.pgettext('organization', 'club', 'clubs', 2), 'les clubs')
    self.assertEqual(locale.pgettext('stick', 'club', 'clubs', 1), 'le bâton')
    self.assertEqual(locale.pgettext('stick', 'club', 'clubs', 2), 'les bâtons')