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
def test_volume_attachment_delete(self):
    fake_server = server.Server(id=str(uuid.uuid4()))
    fake_volume = volume.Volume(id=str(uuid.uuid4()))
    self.verify_delete(self.proxy.delete_volume_attachment, volume_attachment.VolumeAttachment, ignore_missing=False, method_args=[fake_server, fake_volume], method_kwargs={}, expected_args=[], expected_kwargs={'id': fake_volume.id, 'server_id': fake_server.id})