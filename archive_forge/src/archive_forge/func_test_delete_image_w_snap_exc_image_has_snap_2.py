import hashlib
import io
from unittest import mock
from oslo_utils.secretutils import md5
from oslo_utils import units
from glance_store._drivers import rbd as rbd_store
from glance_store import exceptions
from glance_store import location as g_location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
from glance_store.tests import utils as test_utils
def test_delete_image_w_snap_exc_image_has_snap_2(self):

    def _fake_remove(*args, **kwargs):
        self.called_commands_actual.append('remove')
        raise MockRBD.ImageHasSnapshots()
    mock.patch.object(MockRBD.RBD, 'trash_move', side_effect=MockRBD.ImageBusy).start()
    with mock.patch.object(MockRBD.RBD, 'remove') as remove:
        remove.side_effect = _fake_remove
        self.assertRaises(exceptions.InUseByStore, self.store._delete_image, 'fake_pool', self.location.image)
        self.called_commands_expected = ['remove']
    MockRBD.RBD.trash_move.assert_called_once_with(mock.ANY, 'fake_image')