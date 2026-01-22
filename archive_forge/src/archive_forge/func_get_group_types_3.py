from datetime import datetime
from cinderclient.tests.unit import fakes
from cinderclient.tests.unit.v3 import fakes_base
from cinderclient.v3 import client
def get_group_types_3(self, **kw):
    return (200, {}, {'group_type': {'id': 3, 'name': 'test-type-3', 'description': 'test_type-3-desc', 'group_specs': {}, 'is_public': False}})