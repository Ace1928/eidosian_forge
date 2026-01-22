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
def test_create_image_conf_features(self):
    fsid = 'fake'
    features = '3'
    conf_get_mock = mock.Mock(return_value=features)
    conn = mock.Mock(conf_get=conf_get_mock)
    ioctxt = mock.sentinel.ioctxt
    name = '1'
    size = 1024
    order = 3
    with mock.patch.object(rbd_store.rbd.RBD, 'create') as create_mock:
        location = self.store._create_image(fsid, conn, ioctxt, name, size, order)
        self.assertEqual(fsid, location.specs['fsid'])
        self.assertEqual(rbd_store.DEFAULT_POOL, location.specs['pool'])
        self.assertEqual(name, location.specs['image'])
        self.assertEqual(rbd_store.DEFAULT_SNAPNAME, location.specs['snapshot'])
    create_mock.assert_called_once_with(ioctxt, name, size, order, old_format=False, features=3)