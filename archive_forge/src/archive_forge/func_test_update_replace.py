from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine import resource
from heat.engine.resources.openstack.manila import security_service
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_update_replace(self):
    ss = self._create_resource('security_service', self.rsrc_defn, self.stack)
    t = template_format.parse(stack_template_update_replace)
    rsrc_defns = template.Template(t).resource_definitions(self.stack)
    new_ss = rsrc_defns['security_service']
    self.assertEqual(0, self.client.security_services.update.call_count)
    err = self.assertRaises(resource.UpdateReplace, scheduler.TaskRunner(ss.update, new_ss))
    msg = 'The Resource security_service requires replacement.'
    self.assertEqual(msg, str(err))