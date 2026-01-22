import contextlib
import datetime
from unittest import mock
import uuid
import warnings
from openstack.block_storage.v3 import volume
from openstack.compute.v2 import _proxy
from openstack.compute.v2 import aggregate
from openstack.compute.v2 import availability_zone as az
from openstack.compute.v2 import extension
from openstack.compute.v2 import flavor
from openstack.compute.v2 import hypervisor
from openstack.compute.v2 import image
from openstack.compute.v2 import keypair
from openstack.compute.v2 import migration
from openstack.compute.v2 import quota_set
from openstack.compute.v2 import server
from openstack.compute.v2 import server_action
from openstack.compute.v2 import server_group
from openstack.compute.v2 import server_interface
from openstack.compute.v2 import server_ip
from openstack.compute.v2 import server_migration
from openstack.compute.v2 import server_remote_console
from openstack.compute.v2 import service
from openstack.compute.v2 import usage
from openstack.compute.v2 import volume_attachment
from openstack import resource
from openstack.tests.unit import test_proxy_base
from openstack import warnings as os_warnings
def test_volume_attachment_create__legacy_parameters(self):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        self.verify_create(self.proxy.create_volume_attachment, volume_attachment.VolumeAttachment, method_kwargs={'server': 'server_id', 'volumeId': 'volume_id'}, expected_kwargs={'server_id': 'server_id', 'volume_id': 'volume_id'})
        self.assertEqual(1, len(w))
        self.assertEqual(os_warnings.OpenStackDeprecationWarning, w[-1].category)
        self.assertIn('This method was called with a volume_id or volumeId argument', str(w[-1]))