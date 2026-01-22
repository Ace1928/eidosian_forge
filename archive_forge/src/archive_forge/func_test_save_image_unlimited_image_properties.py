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
def test_save_image_unlimited_image_properties(self):
    self.config(image_property_quota=-1)
    self.image.extra_properties = {'foo': 'bar'}
    self.image_repo_proxy.save(self.image)
    self.image_repo_mock.save.assert_called_once_with(self.base_image, from_state=None)