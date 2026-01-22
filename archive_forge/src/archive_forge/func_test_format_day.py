import datetime
import os
import shutil
import tempfile
import unittest
import tornado.locale
from tornado.escape import utf8, to_unicode
from tornado.util import unicode_type
def test_format_day(self):
    locale = tornado.locale.get('en_US')
    date = datetime.datetime(2013, 4, 28, 18, 35)
    self.assertEqual(locale.format_day(date=date, dow=True), 'Sunday, April 28')
    self.assertEqual(locale.format_day(date=date, dow=False), 'April 28')