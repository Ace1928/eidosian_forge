from heat.engine import node_data
from heat.tests import common
def make_test_data():
    return {'id': 42, 'name': 'foo', 'reference_id': 'foo-000000', 'attrs': {'foo': 'bar', ('foo', 'bar', 'baz'): 'quux', ('blarg', 'wibble'): 'foo'}, 'action': 'CREATE', 'status': 'COMPLETE', 'uuid': '000000-0000-0000-0000000'}