from tensorflow.python.eager import context
from tensorflow.python.eager import execute
def clear_op_callbacks():
    """Clear all op callbacks registered in the current thread."""
    for callback in context.context().op_callbacks:
        remove_op_callback(callback)