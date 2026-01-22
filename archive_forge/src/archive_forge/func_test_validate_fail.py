from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.mistral import cron_trigger
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_validate_fail(self):
    t = template_format.parse(stack_template)
    del t['resources']['cron_trigger']['properties']['first_time']
    del t['resources']['cron_trigger']['properties']['pattern']
    stack = utils.parse_stack(t)
    resource_defns = stack.t.resource_definitions(stack)
    self.rsrc_defn = resource_defns['cron_trigger']
    ct = self._create_resource('trigger', self.rsrc_defn, self.stack)
    msg = 'At least one of the following properties must be specified: pattern, first_time'
    self.assertRaisesRegex(exception.PropertyUnspecifiedError, msg, ct.validate)