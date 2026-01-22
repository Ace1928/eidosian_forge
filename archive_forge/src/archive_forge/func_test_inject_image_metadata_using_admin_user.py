import os
from unittest import mock
import glance_store
from oslo_config import cfg
import glance.async_.flows.api_image_import as import_flow
import glance.async_.flows.plugins.inject_image_metadata as inject_metadata
from glance.common import utils
from glance import domain
from glance import gateway
from glance.tests.unit import utils as test_unit_utils
import glance.tests.utils as test_utils
def test_inject_image_metadata_using_admin_user(self):
    context = test_unit_utils.get_fake_context(roles='admin')
    inject_image_metadata = inject_metadata._InjectMetadataProperties(context, self.task.task_id, self.task_type, self.wrapper)
    self.config(inject={'test': 'abc'}, group='inject_metadata_properties')
    inject_image_metadata.execute()
    self.img_repo.save.assert_called_once_with(self.image, 'queued')