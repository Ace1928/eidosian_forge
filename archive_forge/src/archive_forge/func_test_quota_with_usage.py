import fixtures
import http.client as http
from oslo_utils import units
from glance.quota import keystone as ks_quota
from glance.tests import functional
from glance.tests.functional.v2.test_images import get_enforcer_class
from glance.tests import utils as test_utils
def test_quota_with_usage(self):
    self.set_limit({'image_size_total': 5, 'image_count_total': 10, 'image_stage_total': 15, 'image_count_uploading': 20})
    self.start_server()
    expected = {'image_size_total': {'limit': 5, 'usage': 0}, 'image_count_total': {'limit': 10, 'usage': 0}, 'image_stage_total': {'limit': 15, 'usage': 0}, 'image_count_uploading': {'limit': 20, 'usage': 0}}
    self._assert_usage(expected)
    data = test_utils.FakeData(1 * units.Mi)
    image_id = self._create_and_stage(data_iter=data)
    expected['image_count_uploading']['usage'] = 1
    expected['image_count_total']['usage'] = 1
    expected['image_stage_total']['usage'] = 1
    self._assert_usage(expected)
    self._import_direct(image_id, ['store1'])
    self._assert_usage(expected)
    self._wait_for_import(image_id)
    expected['image_count_uploading']['usage'] = 0
    expected['image_stage_total']['usage'] = 0
    expected['image_size_total']['usage'] = 1
    self._assert_usage(expected)
    data = test_utils.FakeData(1 * units.Mi)
    image_id = self._create_and_upload(data_iter=data)
    expected['image_count_total']['usage'] = 2
    expected['image_size_total']['usage'] = 2
    self._assert_usage(expected)
    self.api_delete('/v2/images/%s' % image_id)
    expected['image_count_total']['usage'] = 1
    expected['image_size_total']['usage'] = 1
    self._assert_usage(expected)