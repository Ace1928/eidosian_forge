import copy
from unittest import mock
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import magnum as mc
from heat.engine.clients.os import nova
from heat.engine import resource
from heat.engine.resources.openstack.magnum import cluster
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_cluster_create_unknown_status(self):
    b = self._create_resource('cluster', self.rsrc_defn, self.stack, stat='CREATE_FOO')
    exc = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(b.create))
    self.assertIn('Unknown status creating Cluster', str(exc))