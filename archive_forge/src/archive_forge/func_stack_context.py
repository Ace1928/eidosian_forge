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
def stack_context(stack_name, create_res=True, convergence=False):
    """Decorator for creating and deleting stack.

    Decorator which creates a stack by using the test case's context and
    deletes it afterwards to ensure tests clean up their stacks regardless
    of test success/failure.
    """

    def stack_delete(test_fn):

        @functools.wraps(test_fn)
        def wrapped_test(test_case, *args, **kwargs):

            def create_stack():
                ctx = getattr(test_case, 'ctx', None)
                if ctx is not None:
                    stack = setup_stack_with_mock(test_case, stack_name, ctx, create_res, convergence)
                    setattr(test_case, 'stack', stack)

            def delete_stack():
                stack = getattr(test_case, 'stack', None)
                if stack is not None and stack.id is not None:
                    clean_up_stack(test_case, stack, delete_res=create_res)
            create_stack()
            try:
                test_fn(test_case, *args, **kwargs)
            except Exception as err:
                try:
                    delete_stack()
                finally:
                    raise err from None
            else:
                delete_stack()
        return wrapped_test
    return stack_delete