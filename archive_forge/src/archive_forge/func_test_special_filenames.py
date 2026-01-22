from tornado.httputil import (
from tornado.escape import utf8, native_str
from tornado.log import gen_log
from tornado.testing import ExpectLog
from tornado.test.util import ignore_deprecation
import copy
import datetime
import logging
import pickle
import time
import urllib.parse
import unittest
from typing import Tuple, Dict, List
def test_special_filenames(self):
    filenames = ['a;b.txt', 'a"b.txt', 'a";b.txt', 'a;"b.txt', 'a";";.txt', 'a\\"b.txt', 'a\\b.txt']
    for filename in filenames:
        logging.debug('trying filename %r', filename)
        str_data = '--1234\nContent-Disposition: form-data; name="files"; filename="%s"\n\nFoo\n--1234--' % filename.replace('\\', '\\\\').replace('"', '\\"')
        data = utf8(str_data.replace('\n', '\r\n'))
        args, files = form_data_args()
        parse_multipart_form_data(b'1234', data, args, files)
        file = files['files'][0]
        self.assertEqual(file['filename'], filename)
        self.assertEqual(file['body'], b'Foo')