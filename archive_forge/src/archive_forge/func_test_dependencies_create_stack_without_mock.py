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
def test_dependencies_create_stack_without_mock(self):
    self.stack.store()
    self.current_resources = self.stack._update_or_store_resources()
    self.stack._compute_convg_dependencies(self.stack.ext_rsrcs_db, self.stack.dependencies, self.current_resources)
    self.assertEqual([((1, True), (3, True)), ((2, True), (3, True)), ((3, True), (4, True)), ((3, True), (5, True))], sorted(self.stack._convg_deps._graph.edges()))