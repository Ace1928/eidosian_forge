import sys
from unittest import mock
import urllib.error
from glance_store import exceptions as store_exceptions
from oslo_config import cfg
from oslo_utils import units
import taskflow
import glance.async_.flows.api_image_import as import_flow
from glance.common import exception
from glance.common.scripts.image_import import main as image_import
from glance import context
from glance.domain import ExtraProperties
from glance import gateway
import glance.tests.utils as test_utils
from cursive import exception as cursive_exception
def test_revert_rollback_metadata_value(self):
    action = self.wrapper.__enter__.return_value
    task = import_flow._ImportMetadata(TASK_ID1, TASK_TYPE, self.context, self.wrapper, self.import_req)
    task.properties = {'prop1': 'value1', 'prop2': 'value2'}
    task.old_properties = {'prop1': 'orig_val', 'old_prop': 'old_value'}
    task.old_attributes = {'container_format': 'bare', 'disk_format': 'qcow2'}
    task.revert(None)
    action.set_image_attribute.assert_called_once_with(status='queued', container_format='bare', disk_format='qcow2')
    action.pop_extra_property.assert_called_once_with('prop2')
    action.set_image_extra_properties.assert_called_once_with(task.old_properties)