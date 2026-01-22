import functools
from heat.common import template_format
from heat.engine.clients.os import glance
from heat.engine.clients.os import keystone
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine.resources.aws.ec2 import instance as instances
from heat.engine import stack as parser
from heat.engine import template as templatem
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def setup_stack_with_mock(test_case, stack_name, ctx, create_res=True, convergence=False):
    stack = get_stack(stack_name, ctx, convergence=convergence)
    stack.store()
    if create_res:
        fc = setup_mocks_with_mock(test_case, stack)
        stack.create()
        stack._persist_state()
        validate_setup_mocks_with_mock(stack, fc)
    return stack