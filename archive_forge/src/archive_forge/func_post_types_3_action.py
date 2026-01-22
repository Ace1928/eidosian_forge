from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def post_types_3_action(self, body, **kw):
    _body = None
    resp = 202
    assert len(list(body)) == 1
    action = list(body)[0]
    if action == 'addProjectAccess':
        assert 'project' in body['addProjectAccess']
    elif action == 'removeProjectAccess':
        assert 'project' in body['removeProjectAccess']
    else:
        raise AssertionError('Unexpected action: %s' % action)
    return (resp, {}, _body)