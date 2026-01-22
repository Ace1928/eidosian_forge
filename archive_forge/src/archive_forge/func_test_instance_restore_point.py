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
def test_instance_restore_point(self):
    t = template_format.parse(db_template)
    t['Resources']['MySqlCloudDB']['Properties']['restore_point'] = '1234'
    instance = self._setup_test_instance('dbinstance_create', t)
    self.client.flavors.get.side_effect = [troveexc.NotFound()]
    self.client.flavors.find.return_value = FakeFlavor(1, '1GB')
    scheduler.TaskRunner(instance.create)()
    self.assertEqual((instance.CREATE, instance.COMPLETE), instance.state)
    users = [{'name': 'testuser', 'password': 'pass', 'host': '%', 'databases': [{'name': 'validdb'}]}]
    databases = [{'collate': 'utf8_general_ci', 'character_set': 'utf8', 'name': 'validdb'}]
    self.client.instances.create.assert_called_once_with('test', '1', volume={'size': 30}, databases=databases, users=users, restorePoint={'backupRef': '1234'}, availability_zone=None, datastore='SomeDStype', datastore_version='MariaDB-5.5', nics=[], replica_of=None, replica_count=None)