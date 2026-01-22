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
def get_servers_9015(self, **kw):
    r = {'server': self.get_servers_detail()[2]['servers'][5]}
    r['server']['OS-EXT-AZ:availability_zone'] = 'geneva'
    r['server']['OS-EXT-STS:power_state'] = 0
    flavor = {'disk': 1, 'ephemeral': 0, 'original_name': 'm1.tiny', 'ram': 512, 'swap': 0, 'vcpus': 1, 'extra_specs': {}}
    image = {'id': 'c99d7632-bd66-4be9-aed5-3dd14b223a76'}
    r['server']['image'] = image
    r['server']['flavor'] = flavor
    r['server']['user_id'] = 'fake'
    return (200, {}, r)