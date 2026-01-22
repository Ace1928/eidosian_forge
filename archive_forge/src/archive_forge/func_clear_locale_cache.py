import datetime
import os
import shutil
import tempfile
import unittest
import tornado.locale
from tornado.escape import utf8, to_unicode
from tornado.util import unicode_type
def clear_locale_cache(self):
    tornado.locale.Locale._cache = {}