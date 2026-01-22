import json
import posixpath
from io import BytesIO, StringIO
from time import time
from unittest import skipIf
from dulwich.tests import TestCase
from ..objects import Blob, Commit, Tag, Tree, parse_timezone
from ..tests.test_object_store import ObjectStoreTests
def test_get_object_stat(self):
    with patch('geventhttpclient.HTTPClient.request', lambda *args: Response(headers={'content-length': '10'})):
        self.assertEqual(self.conn.get_object_stat('a')['content-length'], '10')