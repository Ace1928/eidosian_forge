from datetime import datetime
from cinderclient.tests.unit import fakes
from cinderclient.tests.unit.v3 import fakes_base
from cinderclient.v3 import client
def get_groups_5678(self, **kw):
    return (200, {}, {'group': _stub_group(id='5678')})