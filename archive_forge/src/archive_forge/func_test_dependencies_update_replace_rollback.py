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
def test_dependencies_update_replace_rollback(self):
    t = template_format.parse(tools.string_template_five)
    tmpl = templatem.Template(t)
    self.stack.t = tmpl
    self.stack.t.id = 1
    self.stack.prev_raw_template_id = 2
    db_resources = self._fake_db_resources(self.stack)
    db_resources[1].current_template_id = 1
    res = mock.MagicMock()
    res.id = 6
    res.name = 'E'
    res.requires = {3}
    res.replaces = 1
    res.current_template_id = 2
    db_resources[6] = res
    curr_resources = {res.name: res for id, res in db_resources.items()}
    curr_resources['E'] = db_resources[1]
    self.stack._compute_convg_dependencies(db_resources, self.stack.dependencies, curr_resources)
    self.assertEqual([((1, False), (1, True)), ((1, False), (6, False)), ((1, True), (3, True)), ((2, False), (2, True)), ((2, True), (3, True)), ((3, False), (1, False)), ((3, False), (2, False)), ((3, False), (3, True)), ((3, False), (6, False)), ((3, True), (4, True)), ((3, True), (5, True)), ((4, False), (3, False)), ((4, False), (4, True)), ((5, False), (3, False)), ((5, False), (5, True))], sorted(self.stack._convg_deps._graph.edges()))