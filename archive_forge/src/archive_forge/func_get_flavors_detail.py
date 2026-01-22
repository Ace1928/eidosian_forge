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
def get_flavors_detail(self, **kw):
    flavors = {'flavors': [{'id': 1, 'name': '256 MiB Server', 'ram': 256, 'disk': 10, 'OS-FLV-EXT-DATA:ephemeral': 10, 'os-flavor-access:is_public': True, 'links': {}}, {'id': 2, 'name': '512 MiB Server', 'ram': 512, 'disk': 20, 'OS-FLV-EXT-DATA:ephemeral': 20, 'os-flavor-access:is_public': False, 'links': {}}, {'id': 4, 'name': '1024 MiB Server', 'ram': 1024, 'disk': 10, 'OS-FLV-EXT-DATA:ephemeral': 10, 'os-flavor-access:is_public': True, 'links': {}}, {'id': 'aa1', 'name': '128 MiB Server', 'ram': 128, 'disk': 0, 'OS-FLV-EXT-DATA:ephemeral': 0, 'os-flavor-access:is_public': True, 'links': {}}]}
    if 'is_public' not in kw:
        filter_is_public = True
    elif kw['is_public'].lower() == 'none':
        filter_is_public = None
    else:
        filter_is_public = strutils.bool_from_string(kw['is_public'], True)
    if filter_is_public is not None:
        if filter_is_public:
            flavors['flavors'] = [v for v in flavors['flavors'] if v['os-flavor-access:is_public']]
        else:
            flavors['flavors'] = [v for v in flavors['flavors'] if not v['os-flavor-access:is_public']]
    if self.api_version >= api_versions.APIVersion('2.55'):
        for flavor in flavors['flavors']:
            flavor['description'] = None
        new_flavor = copy.deepcopy(flavors['flavors'][0])
        new_flavor['id'] = 'with-description'
        new_flavor['name'] = 'with-description'
        new_flavor['description'] = 'test description'
        flavors['flavors'].append(new_flavor)
    if self.api_version >= api_versions.APIVersion('2.61'):
        for flavor in flavors['flavors']:
            flavor['extra_specs'] = {'test': 'value'}
    return (200, FAKE_RESPONSE_HEADERS, flavors)