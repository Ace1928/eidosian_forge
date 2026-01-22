import collections
import copy
from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.manila import share as mshare
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_allowed_access_type(self):
    tmp = template_format.parse(manila_template)
    properties = tmp['resources']['test_share']['properties']
    properties['access_rules'][0]['access_type'] = 'domain'
    stack = utils.parse_stack(tmp, stack_name='access_type')
    self.assertRaisesRegex(exception.StackValidationFailed, '.* "domain" is not an allowed value', stack.validate)