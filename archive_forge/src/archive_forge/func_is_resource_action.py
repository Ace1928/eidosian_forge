import inspect
import jmespath
from botocore.compat import six
def is_resource_action(action_handle):
    if six.PY3:
        return inspect.isfunction(action_handle)
    else:
        return inspect.ismethod(action_handle)