from datetime import datetime
from cinderclient.tests.unit import fakes
from cinderclient.tests.unit.v3 import fakes_base
from cinderclient.v3 import client
def get_group_types(self, **kw):
    return (200, {}, {'group_types': [{'id': 1, 'name': 'test-type-1', 'description': 'test_type-1-desc', 'group_specs': {}}, {'id': 2, 'name': 'test-type-2', 'description': 'test_type-2-desc', 'group_specs': {}}]})