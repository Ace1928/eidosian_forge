from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.barbican import container
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_create_failed_on_validation(self):
    tmpl = template_format.parse(template_by_name())
    stack = utils.parse_stack(tmpl)
    props = tmpl['resources']['container']['properties']
    props['secrets'].append({'name': 'secret3', 'ref': 'ref1'})
    defn = rsrc_defn.ResourceDefinition('failed_container', 'OS::Barbican::GenericContainer', props)
    res = container.GenericContainer('foo', defn, stack)
    self.assertRaisesRegex(exception.StackValidationFailed, 'Duplicate refs are not allowed', res.validate)