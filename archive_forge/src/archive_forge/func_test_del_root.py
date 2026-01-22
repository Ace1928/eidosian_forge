import json
import posixpath
from io import BytesIO, StringIO
from time import time
from unittest import skipIf
from dulwich.tests import TestCase
from ..objects import Blob, Commit, Tag, Tree, parse_timezone
from ..tests.test_object_store import ObjectStoreTests
def test_del_root(self):
    with patch('dulwich.contrib.swift.SwiftConnector.del_object', lambda *args: None):
        with patch('dulwich.contrib.swift.SwiftConnector.get_container_objects', lambda *args: ({'name': 'a'}, {'name': 'b'})):
            with patch('geventhttpclient.HTTPClient.request', lambda *args: Response()):
                self.assertEqual(self.conn.del_root(), None)