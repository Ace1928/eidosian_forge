from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import nova
from heat.engine.clients.os import sahara
from heat.engine.resources.openstack.sahara import templates as st
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_template_invalid_name(self):
    tmpl = template_format.parse(cluster_template_without_name)
    stack = utils.parse_stack(tmpl)
    ct = stack['cluster_template!']
    self.ct_mgr.create.return_value = self.fake_ct
    scheduler.TaskRunner(ct.create)()
    self.assertEqual((ct.CREATE, ct.COMPLETE), ct.state)
    self.assertEqual(self.fake_ct.id, ct.resource_id)
    name = self.ct_mgr.create.call_args[1]['name']
    self.assertIn('-clustertemplate-', name)