import errno
import io
from unittest import mock
import sys
import uuid
import fixtures
from oslo_config import cfg
from oslo_utils import units
import glance_store as store
from glance_store import exceptions
from glance_store import location
from glance_store.tests import base
from glance_store.tests.unit.cinder import test_cinder_base
from glance_store.tests.unit import test_store_capabilities as test_cap
from glance_store._drivers.cinder import store as cinder # noqa
def test_cinder_add_different_backend(self):
    self.store = cinder.Store(self.conf, backend='cinder2')
    self.store.configure()
    self.register_store_backend_schemes(self.store, 'cinder', 'cinder2')
    fake_volume = mock.MagicMock(id=str(uuid.uuid4()), status='available', size=1)
    volume_file = io.BytesIO()
    self._test_cinder_add(fake_volume, volume_file, backend='cinder2', is_multi_store=True)