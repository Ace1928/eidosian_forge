import copy
import datetime
import re
from unittest import mock
from urllib import parse
from oslo_utils import strutils
import novaclient
from novaclient import api_versions
from novaclient import client as base_client
from novaclient import exceptions
from novaclient.tests.unit import fakes
from novaclient.tests.unit import utils
from novaclient.v2 import client
@classmethod
def check_server_actions(cls, body):
    action = list(body)[0]
    if action == 'reboot':
        assert list(body[action]) == ['type']
        assert body[action]['type'] in ['HARD', 'SOFT']
    elif action == 'resize':
        assert 'flavorRef' in body[action]
    elif action in cls.none_actions:
        assert body[action] is None
    elif action == 'changePassword':
        assert list(body[action]) == ['adminPass']
    elif action in cls.type_actions:
        assert list(body[action]) == ['type']
    elif action == 'os-resetState':
        assert list(body[action]) == ['state']
    elif action == 'resetNetwork':
        assert body[action] is None
    elif action in ['addSecurityGroup', 'removeSecurityGroup']:
        assert list(body[action]) == ['name']
    elif action == 'trigger_crash_dump':
        assert body[action] is None
    else:
        return False
    return True