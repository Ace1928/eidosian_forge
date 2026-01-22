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
def test_instance_create_fails(self):
    cfg.CONF.set_override('action_retry_limit', 0)
    t = template_format.parse(db_template)
    instance = self._setup_test_instance('dbinstance_create', t)
    self.fake_instance.status = 'ERROR'
    self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(instance.create))
    self.fake_instance.status = 'ACTIVE'