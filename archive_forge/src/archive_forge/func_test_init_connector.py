import json
import posixpath
from io import BytesIO, StringIO
from time import time
from unittest import skipIf
from dulwich.tests import TestCase
from ..objects import Blob, Commit, Tag, Tree, parse_timezone
from ..tests.test_object_store import ObjectStoreTests
def test_init_connector(self):
    self.assertEqual(self.conn.auth_ver, '1')
    self.assertEqual(self.conn.auth_url, 'http://127.0.0.1:8080/auth/v1.0')
    self.assertEqual(self.conn.user, 'test:tester')
    self.assertEqual(self.conn.password, 'testing')
    self.assertEqual(self.conn.root, 'fakerepo')
    self.assertEqual(self.conn.storage_url, 'http://127.0.0.1:8080/v1.0/AUTH_fakeuser')
    self.assertEqual(self.conn.token, '12' * 10)
    self.assertEqual(self.conn.http_timeout, 1)
    self.assertEqual(self.conn.http_pool_length, 1)
    self.assertEqual(self.conn.concurrency, 1)
    self.conf.set('swift', 'auth_ver', '2')
    self.conf.set('swift', 'auth_url', 'http://127.0.0.1:8080/auth/v2.0')
    with patch('geventhttpclient.HTTPClient.request', fake_auth_request_v2):
        conn = swift.SwiftConnector('fakerepo', conf=self.conf)
    self.assertEqual(conn.user, 'tester')
    self.assertEqual(conn.tenant, 'test')
    self.conf.set('swift', 'auth_ver', '1')
    self.conf.set('swift', 'auth_url', 'http://127.0.0.1:8080/auth/v1.0')
    with patch('geventhttpclient.HTTPClient.request', fake_auth_request_v1_error):
        self.assertRaises(swift.SwiftException, lambda: swift.SwiftConnector('fakerepo', conf=self.conf))