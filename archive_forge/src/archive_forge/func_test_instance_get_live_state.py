from unittest import mock
import uuid
from oslo_config import cfg
from troveclient import exceptions as troveexc
from troveclient.v1 import users
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import trove
from heat.engine import resource
from heat.engine.resources.openstack.trove import instance as dbinstance
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template as tmpl
from heat.tests import common
from heat.tests import utils
def test_instance_get_live_state(self):
    self.fake_instance.to_dict = mock.Mock(return_value={'name': 'test_instance', 'flavor': {'id': '1'}, 'volume': {'size': 30}})
    fake_db1 = mock.Mock()
    fake_db1.name = 'validdb'
    fake_db2 = mock.Mock()
    fake_db2.name = 'secondvaliddb'
    self.client.databases.list.return_value = [fake_db1, fake_db2]
    expected = {'flavor': '1', 'name': 'test_instance', 'size': 30, 'databases': [{'name': 'validdb', 'character_set': 'utf8', 'collate': 'utf8_general_ci'}, {'name': 'secondvaliddb'}]}
    t = template_format.parse(db_template)
    instance = self._setup_test_instance('get_live_state_test', t)
    reality = instance.get_live_state(instance.properties)
    self.assertEqual(expected, reality)