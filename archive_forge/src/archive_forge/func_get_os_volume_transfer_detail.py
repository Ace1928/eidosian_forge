from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def get_os_volume_transfer_detail(self, **kw):
    base_uri = 'http://localhost:8776'
    tenant_id = '0fa851f6668144cf9cd8c8419c1646c1'
    transfer1 = '5678'
    transfer2 = 'f625ec3e-13dd-4498-a22a-50afd534cc41'
    return (200, {}, {'transfers': [_stub_transfer_full(transfer1, base_uri, tenant_id), _stub_transfer_full(transfer2, base_uri, tenant_id)]})