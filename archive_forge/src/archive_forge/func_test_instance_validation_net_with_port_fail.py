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
def test_instance_validation_net_with_port_fail(self):
    t = template_format.parse(db_template)
    t['Resources']['MySqlCloudDB']['Properties']['networks'] = [{'port': 'someportuuid', 'network': 'somenetuuid'}]
    instance = self._setup_test_instance('dbinstance_test', t)
    self._stubout_validate(instance, neutron=True, mock_net_constraint=True)
    ex = self.assertRaises(exception.StackValidationFailed, instance.validate)
    self.assertEqual('Either network or port must be provided.', str(ex))