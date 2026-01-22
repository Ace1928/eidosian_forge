import io
from unittest import mock
from oslo_config import cfg
from oslo_utils import units
import glance_store as store
from glance_store._drivers import rbd as rbd_store
from glance_store import exceptions
from glance_store import location as g_location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
def test_delete_image_snap_has_external_references(self):
    with mock.patch.object(MockRBD.Image, 'list_children') as mocked:
        mocked.return_value = True
        self.store._delete_image('fake_pool', self.location.image, snapshot_name='snap')