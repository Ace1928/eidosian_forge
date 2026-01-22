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
@mock.patch('oslo_utils.eventletutils.is_monkey_patched')
def test_delete_image_in_native_thread(self, mock_patched):
    mock_patched.return_value = True
    fake_proxy = mock.MagicMock()
    fake_rbd = mock.MagicMock()
    fake_ioctx = mock.MagicMock()
    with mock.patch.object(rbd_store.tpool, 'Proxy') as tpool_mock, mock.patch.object(rbd_store.rbd, 'RBD') as rbd_mock, mock.patch.object(self.store, 'get_connection') as mock_conn:
        mock_get_conn = mock_conn.return_value.__enter__.return_value
        mock_ioctx = mock_get_conn.open_ioctx.return_value.__enter__
        mock_ioctx.return_value = fake_ioctx
        tpool_mock.return_value = fake_proxy
        rbd_mock.return_value = fake_rbd
        self.store._delete_image('fake_pool', self.location.image)
        tpool_mock.assert_called_once_with(fake_rbd)
        fake_proxy.remove.assert_called_once_with(fake_ioctx, self.location.image)