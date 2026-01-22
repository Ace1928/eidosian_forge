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
def test_image_stage_fail_does_not_set_host(self):
    self.config(public_endpoint='http://worker1.example.com')
    image_id = str(uuid.uuid4())
    request = unit_test_utils.get_fake_request(roles=['admin', 'member'])
    image = FakeImage(image_id=image_id)
    self.image_repo.result = image
    exc_cls = glance_store.exceptions.StorageFull
    with mock.patch.object(filesystem.Store, 'add', side_effect=exc_cls):
        self.assertRaises(webob.exc.HTTPRequestEntityTooLarge, self.controller.stage, request, image_id, 'YYYY', 4)
    self.assertNotIn('os_glance_stage_host', image.extra_properties)