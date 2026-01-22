from unittest import mock
from novaclient import client as base_client
from novaclient import exceptions as nova_exceptions
import requests
from urllib import parse as urlparse
from heat.tests import fakes
def post_servers_5678_action(self, body, **kw):
    _body = None
    resp = 202
    if len(body.keys()) != 1:
        raise AssertionError('No action in body')
    action = next(iter(body))
    if action in ['addFloatingIp', 'removeFloatingIp']:
        keys = list(body[action].keys())
        if keys != ['address']:
            raise AssertionError('Unexpection action keys for %s: %s' % (action, keys))
    return (resp, _body)