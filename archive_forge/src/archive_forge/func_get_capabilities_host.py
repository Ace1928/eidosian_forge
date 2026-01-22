from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def get_capabilities_host(self, **kw):
    return (200, {}, {'namespace': 'OS::Storage::Capabilities::fake', 'vendor_name': 'OpenStack', 'volume_backend_name': 'lvm', 'pool_name': 'pool', 'storage_protocol': 'iSCSI', 'properties': {'compression': {'title': 'Compression', 'description': 'Enables compression.', 'type': 'boolean'}}})