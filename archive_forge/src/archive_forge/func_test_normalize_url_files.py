import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_normalize_url_files(self):
    normalize_url = urlutils.normalize_url

    def norm_file(expected, path):
        url = normalize_url(path)
        self.assertStartsWith(url, 'file:///')
        if sys.platform == 'win32':
            url = url[len('file:///C:'):]
        else:
            url = url[len('file://'):]
        self.assertEndsWith(url, expected)
    norm_file('path/to/foo', 'path/to/foo')
    norm_file('/path/to/foo', '/path/to/foo')
    norm_file('path/to/foo', '../path/to/foo')
    try:
        'uni/µ'.encode(osutils.get_user_encoding())
    except UnicodeError:
        pass
    else:
        norm_file('uni/%C2%B5', 'uni/µ')
    norm_file('uni/%25C2%25B5', 'uni/%C2%B5')
    norm_file('uni/%20b', 'uni/ b')
    norm_file('%27%20%3B/%3F%3A%40%26%3D%2B%24%2C%23', "' ;/?:@&=+$,#")