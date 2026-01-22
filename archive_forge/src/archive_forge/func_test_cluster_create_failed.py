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
def test_cluster_create_failed(self):
    cfg.CONF.set_override('action_retry_limit', 0)
    b = self._create_resource('cluster', self.rsrc_defn, self.stack, stat='CREATE_FAILED')
    exc = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(b.create))
    self.assertIn('Failed to create Cluster', str(exc))