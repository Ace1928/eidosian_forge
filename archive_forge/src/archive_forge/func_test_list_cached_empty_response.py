import testtools
from unittest import mock
from glanceclient.common import utils as common_utils
from glanceclient import exc
from glanceclient.tests import utils
from glanceclient.v2 import cache
@mock.patch.object(common_utils, 'has_version')
def test_list_cached_empty_response(self, mock_has_version):
    dummy_fixtures = {'/v2/cache': {'GET': ({}, {'cached_images': [], 'queued_images': []})}}
    dummy_api = utils.FakeAPI(dummy_fixtures)
    dummy_controller = cache.Controller(dummy_api)
    mock_has_version.return_value = True
    images = dummy_controller.list()
    self.assertEqual(0, len(images['cached_images']))
    self.assertEqual(0, len(images['queued_images']))