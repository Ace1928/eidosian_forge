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
def test_add_w_image_size_zero_to_different_backend(self):
    """Assert that correct size is returned even though 0 was provided."""
    self.store = rbd_store.Store(self.conf, backend='ceph2')
    self.store.configure()
    self.called_commands_actual = []
    self.called_commands_expected = []
    self.store_specs = {'pool': 'fake_pool_1', 'image': 'fake_image_1', 'snapshot': 'fake_snapshot_1'}
    self.location = rbd_store.StoreLocation(self.store_specs, self.conf)
    self.data_len = 3 * units.Ki
    self.data_iter = io.BytesIO(b'*' * self.data_len)
    self.store.chunk_size = units.Ki
    with mock.patch.object(rbd_store.rbd.Image, 'resize') as resize:
        with mock.patch.object(rbd_store.rbd.Image, 'write') as write:
            ret = self.store.add('fake_image_id', self.data_iter, 0)
            self.assertTrue(resize.called)
            self.assertTrue(write.called)
            self.assertEqual(ret[1], self.data_len)
            self.assertEqual('ceph2', ret[3]['store'])