from datetime import datetime
from cinderclient.tests.unit import fakes
from cinderclient.tests.unit.v3 import fakes_base
from cinderclient.v3 import client
def post_groups_action(self, body, **kw):
    group = _stub_group(id='1234', group_type='my_group_type', volume_types=['type1', 'type2'])
    resp = 202
    assert len(list(body)) == 1
    action = list(body)[0]
    if action == 'create-from-src':
        assert 'group_snapshot_id' in body[action] or 'source_group_id' in body[action]
    else:
        raise AssertionError('Unexpected action: %s' % action)
    return (resp, {}, {'group': group})