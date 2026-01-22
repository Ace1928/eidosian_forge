import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def post_shares_1234_action(self, body, **kw):
    _body = None
    resp = 202
    assert len(list(body)) == 1
    action = list(body)[0]
    if action in ('os-allow_access', 'allow_access'):
        expected = ['access_to', 'access_type']
        actual = sorted(list(body[action]))
        err_msg = "expected '%s', actual is '%s'" % (expected, actual)
        assert expected == actual, err_msg
        _body = {'access': {}}
    elif action in ('os-deny_access', 'deny_access'):
        assert list(body[action]) == ['access_id']
    elif action in ('os-access_list', 'access_list'):
        assert body[action] is None
    elif action in ('os-reset_status', 'reset_status'):
        assert 'status' in body.get('reset_status', body.get('os-reset_status'))
    elif action in ('os-force_delete', 'force_delete'):
        assert body[action] is None
    elif action in ('os-extend', 'os-shrink', 'extend', 'shrink'):
        assert body[action] is not None
        assert body[action]['new_size'] is not None
    elif action in ('unmanage',):
        assert body[action] is None
    elif action in ('revert',):
        assert body[action] is not None
        assert body[action]['snapshot_id'] is not None
    elif action in ('migration_cancel', 'migration_complete', 'migration_get_progress'):
        assert body[action] is None
        if 'migration_get_progress' == action:
            _body = {'total_progress': 50}
            return (200, {}, _body)
    elif action in ('os-migrate_share', 'migrate_share', 'migration_start'):
        assert 'host' in body[action]
    elif action == 'reset_task_state':
        assert 'task_state' in body[action]
    elif action in ('soft_delete', 'restore'):
        assert body[action] is None
    else:
        raise AssertionError('Unexpected share action: %s' % action)
    return (resp, {}, _body)