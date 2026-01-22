from datetime import datetime
from cinderclient.tests.unit import fakes
from cinderclient.tests.unit.v3 import fakes_base
from cinderclient.v3 import client
def get_volume_transfers_5678(self, **kw):
    base_uri = 'http://localhost:8776'
    tenant_id = '0fa851f6668144cf9cd8c8419c1646c1'
    transfer1 = '5678'
    return (200, {}, {'transfer': fakes_base._stub_transfer_full(transfer1, base_uri, tenant_id)})