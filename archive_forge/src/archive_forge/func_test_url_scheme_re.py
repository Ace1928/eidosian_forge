import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_url_scheme_re(self):

    def test_one(url, scheme_and_path):
        """Assert that _url_scheme_re correctly matches

            :param scheme_and_path: The (scheme, path) that should be matched
                can be None, to indicate it should not match
            """
        m = urlutils._url_scheme_re.match(url)
        if scheme_and_path is None:
            self.assertEqual(None, m)
        else:
            self.assertEqual(scheme_and_path[0], m.group('scheme'))
            self.assertEqual(scheme_and_path[1], m.group('path'))
    test_one('/path', None)
    test_one('C:/path', None)
    test_one('../path/to/foo', None)
    test_one('../path/to/foå', None)
    test_one('http://host/path/', ('http', 'host/path/'))
    test_one('sftp://host/path/to/foo', ('sftp', 'host/path/to/foo'))
    test_one('file:///usr/bin', ('file', '/usr/bin'))
    test_one('file:///C:/Windows', ('file', '/C:/Windows'))
    test_one('file:///C|/Windows', ('file', '/C|/Windows'))
    test_one('readonly+sftp://host/path/å', ('readonly+sftp', 'host/path/å'))
    test_one('/path/to/://foo', None)
    test_one('scheme:stuff://foo', ('scheme', 'stuff://foo'))
    test_one('C://foo', None)
    test_one('ab://foo', ('ab', 'foo'))