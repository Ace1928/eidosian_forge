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
@mock.patch.object(filesystem.Store, 'add')
def test_image_stage_invalid_image_transition(self, mock_store_add):
    image_id = str(uuid.uuid4())
    request = unit_test_utils.get_fake_request(roles=['admin', 'member'])
    image = FakeImage(image_id=image_id)
    self.image_repo.result = image
    with mock.patch.object(filesystem.Store, 'add') as mock_add:
        mock_add.return_value = ('foo://bar', 4, 'ident', {})
        self.controller.stage(request, image_id, 'YYYY', 4)
    self.assertEqual('uploading', image.status)
    self.assertEqual(4, image.size)
    mock_store_add.side_effect = exception.InvalidImageStatusTransition(cur_status='uploading', new_status='uploading')
    self.assertRaises(webob.exc.HTTPConflict, self.controller.stage, request, image_id, 'YYYY', 4)