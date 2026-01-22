import collections
from unittest import mock
import uuid
from novaclient import client as nc
from novaclient import exceptions as nova_exceptions
from oslo_config import cfg
from oslo_serialization import jsonutils as json
import requests
from heat.common import exception
from heat.engine.clients.os import nova
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_build_userdata_with_ignition(self):
    metadata = {'os-collect-config': {'heat': {'password': '***'}}}
    userdata = '{"ignition": {"version": "3.0"}, "storage": {"files": []}}'
    ud_format = 'SOFTWARE_CONFIG'
    data = self.nova_plugin.build_userdata(metadata, userdata=userdata, user_data_format=ud_format)
    ig = json.loads(data)
    self.assertEqual('/var/lib/heat-cfntools/cfn-init-data', ig['storage']['files'][0]['path'])
    self.assertEqual('/var/lib/cloud/data/cfn-init-data', ig['storage']['files'][1]['path'])
    self.assertEqual('data:,%7B%22os-collect-config%22%3A%20%7B%22heat%22%3A%20%7B%22password%22%3A%20%22%2A%2A%2A%22%7D%7D%7D', ig['storage']['files'][0]['contents']['source'])