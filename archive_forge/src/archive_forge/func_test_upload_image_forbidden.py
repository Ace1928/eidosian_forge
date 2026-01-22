import http.client as http
import io
from unittest import mock
import uuid
from cursive import exception as cursive_exception
import glance_store
from glance_store._drivers import filesystem
from oslo_config import cfg
import webob
import glance.api.policy
import glance.api.v2.image_data
from glance.common import exception
from glance.common import wsgi
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
@mock.patch.object(glance.api.policy.Enforcer, 'enforce')
def test_upload_image_forbidden(self, mock_enforce):
    request = unit_test_utils.get_fake_request()
    image = FakeImage('abcd', owner='tenant1')
    self.image_repo.result = image
    mock_enforce.side_effect = [exception.Forbidden, lambda *a: None]
    self.assertRaises(webob.exc.HTTPForbidden, self.controller.upload, request, unit_test_utils.UUID2, 'YYYY', 4)
    expected_call = [mock.call(mock.ANY, 'upload_image', mock.ANY), mock.call(mock.ANY, 'get_image', mock.ANY)]
    mock_enforce.assert_has_calls(expected_call)