import json
import posixpath
from io import BytesIO, StringIO
from time import time
from unittest import skipIf
from dulwich.tests import TestCase
from ..objects import Blob, Commit, Tag, Tree, parse_timezone
from ..tests.test_object_store import ObjectStoreTests
def test_put_object(self):
    with patch('geventhttpclient.HTTPClient.request', lambda *args, **kwargs: Response()):
        self.assertEqual(self.conn.put_object('a', BytesIO(b'content')), None)