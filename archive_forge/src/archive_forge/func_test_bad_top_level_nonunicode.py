from unittest import mock
from glance_store import backend
from glance_store import exceptions
from glance_store.tests import base
def test_bad_top_level_nonunicode(self):
    metadata = {'key': b'a string'}
    self._bad_metadata(metadata)