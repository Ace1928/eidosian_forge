from unittest import mock
import uuid
from designateclient import exceptions
from designateclient.tests import base
from designateclient import utils
def test_find_resourceid_with_nonhyphen_uuid(self):
    expected = str(uuid.uuid4())
    fakeid = expected.replace('-', '')
    observed = self._find_resourceid_by_name_or_id(fakeid)
    self.assertEqual(expected, observed)