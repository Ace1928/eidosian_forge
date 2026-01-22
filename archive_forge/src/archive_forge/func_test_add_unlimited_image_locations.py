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
def test_add_unlimited_image_locations(self):
    self.config(image_location_quota=-1)
    location1 = {'url': 'file:///fake1.img.tar.gz', 'metadata': {}}
    self.image.locations.append(location1)
    self.assertEqual(1, len(self.image.locations))