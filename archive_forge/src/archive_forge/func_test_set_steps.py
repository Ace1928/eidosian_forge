import json
import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from ironicclient.tests.functional.osc.v1 import base
def test_set_steps(self):
    """Check baremetal deploy template set command for steps.

        Test steps:
        1) Create baremetal deploy template in setUp.
        2) Set steps for deploy template.
        3) Check that baremetal deploy template steps were set.
        """
    steps = [{'interface': 'bios', 'step': 'apply_configuration', 'args': {}, 'priority': 20}]
    self.openstack("baremetal deploy template set --steps '{0}' {1}".format(json.dumps(steps), self.template['uuid']))
    show_prop = self.deploy_template_show(self.template['uuid'], fields=['steps'])
    self.assertEqual(steps, show_prop['steps'])