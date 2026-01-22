from __future__ import absolute_import, division, print_function
import traceback
from functools import wraps
from ansible_collections.community.general.plugins.module_utils.mh.exceptions import ModuleHelperException
def check_mode_skip(func):

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.module.check_mode:
            return func(self, *args, **kwargs)
    return wrapper