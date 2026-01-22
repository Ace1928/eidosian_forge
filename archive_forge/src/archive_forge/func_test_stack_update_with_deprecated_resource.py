import copy
from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.db import api as db_api
from heat.engine.clients.os.keystone import fake_keystoneclient
from heat.engine import environment
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import service
from heat.engine import stack
from heat.engine import support
from heat.engine import template
from heat.objects import stack as stack_object
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def test_stack_update_with_deprecated_resource(self):
    """Test with update deprecated resource to substitute.

        Test checks the following scenario:
        1. Create stack with deprecated resource.
        2. Update stack with substitute resource.
        The test checks that deprecated resource can be update to it's
        substitute resource during update Stack.
        """

    class ResourceTypeB(generic_rsrc.GenericResource):
        count_b = 0

        def update(self, after, before=None, prev_resource=None):
            ResourceTypeB.count_b += 1
            yield
    resource._register_class('ResourceTypeB', ResourceTypeB)

    class ResourceTypeA(ResourceTypeB):
        support_status = support.SupportStatus(status=support.DEPRECATED, message='deprecation_msg', version='2014.2', substitute_class=ResourceTypeB)
        count_a = 0

        def update(self, after, before=None, prev_resource=None):
            ResourceTypeA.count_a += 1
            yield
    resource._register_class('ResourceTypeA', ResourceTypeA)
    TMPL_WITH_DEPRECATED_RES = '\n        heat_template_version: 2015-10-15\n        resources:\n          AResource:\n            type: ResourceTypeA\n        '
    TMPL_WITH_PEPLACE_RES = '\n        heat_template_version: 2015-10-15\n        resources:\n          AResource:\n            type: ResourceTypeB\n        '
    t = template_format.parse(TMPL_WITH_DEPRECATED_RES)
    templ = template.Template(t)
    self.stack = stack.Stack(self.ctx, 'update_test_stack', templ)
    self.stack.store()
    self.stack.create()
    self.assertEqual((stack.Stack.CREATE, stack.Stack.COMPLETE), self.stack.state)
    t = template_format.parse(TMPL_WITH_PEPLACE_RES)
    tmpl2 = template.Template(t)
    updated_stack = stack.Stack(self.ctx, 'updated_stack', tmpl2)
    self.stack.update(updated_stack)
    self.assertEqual((stack.Stack.UPDATE, stack.Stack.COMPLETE), self.stack.state)
    self.assertIn('AResource', self.stack)
    self.assertEqual(1, ResourceTypeB.count_b)
    self.assertEqual(0, ResourceTypeA.count_a)