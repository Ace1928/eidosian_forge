from datetime import datetime
from cinderclient.tests.unit import fakes
from cinderclient.tests.unit.v3 import fakes_base
from cinderclient.v3 import client
def delete_volume_transfers_5678(self, **kw):
    return (202, {}, None)