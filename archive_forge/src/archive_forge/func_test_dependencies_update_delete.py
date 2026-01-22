from datetime import datetime
from datetime import timedelta
from unittest import mock
from oslo_config import cfg
from heat.common import template_format
from heat.engine import environment
from heat.engine import stack as parser
from heat.engine import template as templatem
from heat.objects import raw_template as raw_template_object
from heat.objects import resource as resource_objects
from heat.objects import snapshot as snapshot_objects
from heat.objects import stack as stack_object
from heat.objects import sync_point as sync_point_object
from heat.rpc import worker_client
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
def test_dependencies_update_delete(self):
    tmpl = templatem.Template.create_empty_template(version=self.stack.t.version)
    self.stack.t = tmpl
    self.stack.t.id = 2
    self.stack.prev_raw_template_id = 1
    db_resources = self._fake_db_resources(self.stack)
    curr_resources = {res.name: res for id, res in db_resources.items()}
    self.stack._compute_convg_dependencies(db_resources, self.stack.dependencies, curr_resources)
    self.assertEqual([((3, False), (1, False)), ((3, False), (2, False)), ((4, False), (3, False)), ((5, False), (3, False))], sorted(self.stack._convg_deps._graph.edges()))