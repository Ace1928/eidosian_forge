import errno
import io
from unittest import mock
import sys
import uuid
from oslo_utils import units
from glance_store import exceptions
from glance_store.tests import base
from glance_store.tests.unit.cinder import test_cinder_base
from glance_store.tests.unit import test_store_capabilities
from glance_store._drivers.cinder import store as cinder # noqa
def test_cinder_add_volume_full(self):
    e = IOError()
    volume_file = io.BytesIO()
    e.errno = errno.ENOSPC
    fake_volume = mock.MagicMock(id=str(uuid.uuid4()), status='available', size=1)
    with mock.patch.object(volume_file, 'write', side_effect=e):
        self.assertRaises(exceptions.StorageFull, self._test_cinder_add, fake_volume, volume_file)
    fake_volume.delete.assert_called_once_with()