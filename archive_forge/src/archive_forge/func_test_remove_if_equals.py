import json
import posixpath
from io import BytesIO, StringIO
from time import time
from unittest import skipIf
from dulwich.tests import TestCase
from ..objects import Blob, Commit, Tag, Tree, parse_timezone
from ..tests.test_object_store import ObjectStoreTests
def test_remove_if_equals(self):
    self.fsc.store = self.store
    irc = swift.SwiftInfoRefsContainer(self.fsc, self.object_store)
    irc.remove_if_equals(b'refs/heads/dev', b'cca703b0e1399008b53a1a236d6b4584737649e4')
    self.assertNotIn(b'refs/heads/dev', irc.allkeys())