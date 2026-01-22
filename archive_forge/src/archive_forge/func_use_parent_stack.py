import weakref
from heat.objects import resource as resource_object
def use_parent_stack(parent_proxy, stack):
    parent_proxy._stack_ref = weakref.ref(stack)
    parent_proxy._parent_stack = None