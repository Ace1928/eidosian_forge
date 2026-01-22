from unittest import mock
import uuid
from designateclient import exceptions
from designateclient.tests import base
from designateclient import utils
def test_find_resourceid_with_unique_resource(self):
    observed = self._find_resourceid_by_name_or_id('abcd', by_name=True)
    self.assertEqual('13579bdf-0000-0000-abcd-000000000001', observed)