import json
import posixpath
from io import BytesIO, StringIO
from time import time
from unittest import skipIf
from dulwich.tests import TestCase
from ..objects import Blob, Commit, Tag, Tree, parse_timezone
from ..tests.test_object_store import ObjectStoreTests
def test_init_no_data(self):
    with patch('dulwich.contrib.swift.SwiftConnector', new_callable=create_swift_connector):
        self.assertRaises(Exception, swift.SwiftRepo, 'fakerepo', self.conf)