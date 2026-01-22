import datetime
import http.client as http
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_serialization import jsonutils
import webob
import glance.api.v2.tasks
from glance.common import timeutils
import glance.domain
import glance.gateway
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
@mock.patch('glance.common.scripts.utils.get_image_data_iter')
@mock.patch('glance.common.scripts.utils.validate_location_uri')
def test_create_with_live_time(self, mock_validate_location_uri, mock_get_image_data_iter):
    self.skipTest('Something wrong, this test touches registry')
    request = unit_test_utils.get_fake_request()
    task = {'type': 'import', 'input': {'import_from': 'http://download.cirros-cloud.net/0.3.4/cirros-0.3.4-x86_64-disk.img', 'import_from_format': 'qcow2', 'image_properties': {'disk_format': 'qcow2', 'container_format': 'bare', 'name': 'test-task'}}}
    new_task = self.controller.create(request, task=task)
    executor_factory = self.gateway.get_task_executor_factory(request.context)
    task_executor = executor_factory.new_task_executor(request.context)
    task_executor.begin_processing(new_task.task_id)
    success_task = self.controller.get(request, new_task.task_id)
    task_live_time = success_task.expires_at.replace(second=0, microsecond=0) - success_task.updated_at.replace(second=0, microsecond=0)
    task_live_time_hour = task_live_time.days * 24 + task_live_time.seconds / 3600
    self.assertEqual(CONF.task.task_time_to_live, task_live_time_hour)