import copy
import fixtures
from unittest import mock
from unittest.mock import patch
import uuid
from oslo_limit import exception as ol_exc
from oslo_utils import encodeutils
from oslo_utils import units
from glance.common import exception
from glance.common import store_utils
import glance.quota
from glance.quota import keystone as ks_quota
from glance.tests.unit import fixtures as glance_fixtures
from glance.tests.unit import utils as unit_test_utils
from glance.tests import utils as test_utils
def test_append_location_for_queued_image(self):
    context = FakeContext()
    db_api = unit_test_utils.FakeDB()
    store_api = unit_test_utils.FakeStoreAPI()
    store = unit_test_utils.FakeStoreUtils(store_api)
    base_image = FakeImage()
    base_image.image_id = str(uuid.uuid4())
    image = glance.quota.ImageProxy(base_image, context, db_api, store)
    self.assertIsNone(image.size)
    self.mock_object(store_api, 'get_size_from_backend', unit_test_utils.fake_get_size_from_backend)
    image.locations.append({'url': 'file:///fake.img.tar.gz', 'metadata': {}})
    self.assertIn({'url': 'file:///fake.img.tar.gz', 'metadata': {}}, image.locations)