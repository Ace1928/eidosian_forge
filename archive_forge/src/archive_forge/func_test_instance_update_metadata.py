import copy
from unittest import mock
import uuid
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import cinder
from heat.engine.clients.os import glance
from heat.engine.clients.os import neutron
from heat.engine.clients.os import nova
from heat.engine.clients import progress
from heat.engine import environment
from heat.engine import resource
from heat.engine.resources.aws.ec2 import instance as instances
from heat.engine.resources import scheduler_hints as sh
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_instance_update_metadata(self):
    return_server = self.fc.servers.list()[1]
    instance = self._create_test_instance(return_server, 'ud_md')
    self._stub_glance_for_update()
    ud_tmpl = self._get_test_template('update_stack')[0]
    ud_tmpl.t['Resources']['WebServer']['Metadata'] = {'test': 123}
    resource_defns = ud_tmpl.resource_definitions(instance.stack)
    scheduler.TaskRunner(instance.update, resource_defns['WebServer'])()
    self.assertEqual({'test': 123}, instance.metadata_get())