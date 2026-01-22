import json
from unittest import mock
import glance_store
from oslo_concurrency import processutils
from oslo_config import cfg
from glance.async_.flows import introspect
from glance.async_ import utils as async_utils
from glance import domain
import glance.tests.utils as test_utils
def test_introspect_no_image(self):
    image_create = introspect._Introspect(self.task.task_id, self.task_type, self.img_repo)
    self.task_repo.get.return_value = self.task
    image_id = mock.sentinel.image_id
    image = mock.MagicMock(image_id=image_id, virtual_size=None)
    self.img_repo.get.return_value = image
    with mock.patch.object(processutils, 'execute') as exc_mock:
        exc_mock.return_value = (None, 'some error')
        image_create.execute(image, '/test/path.qcow2')
        self.assertIsNone(image.virtual_size)