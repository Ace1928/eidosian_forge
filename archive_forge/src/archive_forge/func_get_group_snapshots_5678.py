from datetime import datetime
from cinderclient.tests.unit import fakes
from cinderclient.tests.unit.v3 import fakes_base
from cinderclient.v3 import client
def get_group_snapshots_5678(self, **kw):
    return (200, {}, {'group_snapshot': _stub_group_snapshot(id='5678')})