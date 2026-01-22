import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def setcontext(context):
    """Set this thread's context to context."""
    if context in (DefaultContext, BasicContext, ExtendedContext):
        context = context.copy()
        context.clear_flags()
    _current_context_var.set(context)