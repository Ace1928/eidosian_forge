from heat.common import exception
from heat.common import template_format
from heat.engine import resource
from heat.engine.resources.openstack.heat import delay
from heat.engine import status
from heat.tests import common
from heat.tests import utils
from oslo_utils import fixture as utils_fixture
from oslo_utils import timeutils
def test_validate_success(self):
    stk = utils.parse_stack(self.simple_template)
    for res in stk.resources.values():
        self.assertIsNone(res.validate())