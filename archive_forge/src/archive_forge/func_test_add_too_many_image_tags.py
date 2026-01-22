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
def test_add_too_many_image_tags(self):
    self.config(image_tag_quota=1)
    self.image.tags.add('foo')
    exc = self.assertRaises(exception.ImageTagLimitExceeded, self.image.tags.add, 'bar')
    self.assertIn('Attempted: 2, Maximum: 1', encodeutils.exception_to_unicode(exc))