from datetime import datetime
from cinderclient.tests.unit import fakes
from cinderclient.tests.unit.v3 import fakes_base
from cinderclient.v3 import client
def get_groups(self, **kw):
    return (200, {}, {'groups': [_stub_group(detailed=False, id='1234'), _stub_group(detailed=False, id='4567')]})