from __future__ import absolute_import, division, print_function
import traceback
from functools import wraps
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import missing_required_lib
def xcli_wrapper(func):
    """ Catch xcli errors and return a proper message"""

    @wraps(func)
    def wrapper(module, *args, **kwargs):
        try:
            return func(module, *args, **kwargs)
        except errors.CommandExecutionError as e:
            module.fail_json(msg=to_native(e))
    return wrapper