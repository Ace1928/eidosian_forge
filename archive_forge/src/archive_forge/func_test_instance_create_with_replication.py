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
def test_instance_create_with_replication(self):
    t = template_format.parse(db_template_with_replication)
    instance = self._setup_test_instance('dbinstance_test', t)
    scheduler.TaskRunner(instance.create)()
    self.assertEqual((instance.CREATE, instance.COMPLETE), instance.state)
    self.client.instances.create.assert_called_once_with('test', '1', volume={'size': 30}, databases=[], users=[], restorePoint=None, availability_zone=None, datastore=None, datastore_version=None, nics=[], replica_of='0e642916-dd64-43b3-933f-ff34fff69a7f', replica_count=2)