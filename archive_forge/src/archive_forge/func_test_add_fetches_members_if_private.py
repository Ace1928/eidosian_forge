from cryptography import exceptions as crypto_exception
from cursive import exception as cursive_exception
from cursive import signature_utils
import glance_store
from unittest import mock
from glance.common import exception
import glance.location
from glance.tests.unit import base as unit_test_base
from glance.tests.unit import utils as unit_test_utils
from glance.tests import utils
def test_add_fetches_members_if_private(self):
    self.image_stub.locations = [{'url': 'glue', 'metadata': {}, 'status': 'active'}]
    self.image_stub.visibility = 'private'
    self.image_repo.add(self.image)
    self.assertIn('glue', self.store_api.acls)
    acls = self.store_api.acls['glue']
    self.assertFalse(acls['public'])
    self.assertEqual([], acls['write'])
    self.assertEqual([TENANT1, TENANT2], acls['read'])