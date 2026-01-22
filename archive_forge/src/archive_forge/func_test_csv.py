import datetime
import os
import shutil
import tempfile
import unittest
import tornado.locale
from tornado.escape import utf8, to_unicode
from tornado.util import unicode_type
def test_csv(self):
    tornado.locale.load_translations(os.path.join(os.path.dirname(__file__), 'csv_translations'))
    locale = tornado.locale.get('fr_FR')
    self.assertTrue(isinstance(locale, tornado.locale.CSVLocale))
    self.assertEqual(locale.translate('school'), 'école')