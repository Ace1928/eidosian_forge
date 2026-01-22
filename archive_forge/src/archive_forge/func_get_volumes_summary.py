from datetime import datetime
from cinderclient.tests.unit import fakes
from cinderclient.tests.unit.v3 import fakes_base
from cinderclient.v3 import client
def get_volumes_summary(self, **kw):
    return (200, {}, {'volume-summary': {'total_size': 5, 'total_count': 5, 'metadata': {'test_key': ['test_value']}}})