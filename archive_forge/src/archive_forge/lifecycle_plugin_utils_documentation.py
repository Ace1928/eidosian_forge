from oslo_log import log as logging
from heat.engine import resources
Call available post-op methods sequentially.

    In order determined with get_ordinal(), with parameters context, stack,
    current_stack, action, is_stack_failure.
    