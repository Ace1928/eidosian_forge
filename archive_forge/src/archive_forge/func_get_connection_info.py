import sys
from unittest import mock
import ddt
from glance_store._drivers.cinder import base
from glance_store._drivers.cinder import scaleio
from glance_store.tests import base as test_base
from glance_store._drivers.cinder import store as cinder # noqa
from glance_store._drivers.cinder import nfs # noqa
def get_connection_info(self):
    """Return iSCSI connection information"""
    return {'target_discovered': False, 'target_portal': '0.0.0.0:3260', 'target_iqn': 'iqn.2010-10.org.openstack:volume-fake-vol', 'target_lun': 0, 'volume_id': '007dedb8-ddc0-445c-88f1-d07acbe4efcb', 'auth_method': 'CHAP', 'auth_username': '2ttANgVaDRqxtMNK3hUj', 'auth_password': 'fake-password', 'encrypted': False, 'qos_specs': None, 'access_mode': 'rw', 'cacheable': False, 'driver_volume_type': 'iscsi', 'attachment_id': '7f45b2fe-111a-42df-be3e-f02b312ad8ea'}