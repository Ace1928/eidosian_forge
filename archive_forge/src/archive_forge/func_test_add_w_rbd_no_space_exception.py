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
@mock.patch.object(MockRBD.Image, 'resize')
@mock.patch.object(rbd_store.Store, '_create_image')
@mock.patch.object(rbd_store.Store, '_delete_image')
def test_add_w_rbd_no_space_exception(self, delete, create, resize):

    def _fake_create_image(*args, **kwargs):
        self.called_commands_actual.append('create')
        return self.location

    def _fake_delete_image(target_pool, image_name, snapshot_name=None):
        self.assertEqual(self.location.pool, target_pool)
        self.assertEqual(self.location.image, image_name)
        self.assertEqual(self.location.snapshot, snapshot_name)
        self.called_commands_actual.append('delete')

    def _fake_resize(*args, **kwargs):
        raise MockRBD.NoSpace()
    create.side_effect = _fake_create_image
    delete.side_effect = _fake_delete_image
    resize.side_effect = _fake_resize
    self.assertRaises(exceptions.StorageFull, self.store.add, 'fake_image_id', self.data_iter, 0, self.hash_algo)
    self.called_commands_expected = ['create', 'delete']