from datetime import datetime
from cinderclient.tests.unit import fakes
from cinderclient.tests.unit.v3 import fakes_base
from cinderclient.v3 import client
def post_groups_1234_action(self, body, **kw):
    resp = 202
    assert len(list(body)) == 1
    action = list(body)[0]
    if action == 'delete':
        assert 'delete-volumes' in body[action]
    elif action in ('enable_replication', 'disable_replication', 'failover_replication', 'list_replication_targets', 'reset_status'):
        assert action in body
    elif action == 'os-reimage':
        assert 'image_id' in body[action]
    elif action == 'os-extend_volume_completion':
        assert 'error' in body[action]
    else:
        raise AssertionError('Unexpected action: %s' % action)
    return (resp, {}, {})