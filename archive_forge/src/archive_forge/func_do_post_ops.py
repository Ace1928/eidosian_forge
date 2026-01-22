from oslo_log import log as logging
from heat.engine import resources
def do_post_ops(cnxt, stack, current_stack=None, action=None, is_stack_failure=False):
    """Call available post-op methods sequentially.

    In order determined with get_ordinal(), with parameters context, stack,
    current_stack, action, is_stack_failure.
    """
    cinstances = get_plug_point_class_instances()
    if action is None:
        action = stack.action
    _do_ops(cinstances, 'do_post_op', cnxt, stack, current_stack, action, None)