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
def test_delete_image_w_snap_exc_image_busy(self):

    def _fake_unprotect_snap(*args, **kwargs):
        self.called_commands_actual.append('unprotect_snap')
        raise MockRBD.ImageBusy()
    with mock.patch.object(MockRBD.Image, 'unprotect_snap') as mocked:
        mocked.side_effect = _fake_unprotect_snap
        self.assertRaises(exceptions.InUseByStore, self.store._delete_image, 'fake_pool', self.location.image, snapshot_name='snap')
        self.called_commands_expected = ['unprotect_snap']