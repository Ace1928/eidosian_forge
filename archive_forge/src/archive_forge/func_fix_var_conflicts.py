from __future__ import absolute_import, division, print_function
import traceback
from functools import wraps
from ansible_collections.community.general.plugins.module_utils.mh.exceptions import ModuleHelperException
def fix_var_conflicts(output):
    result = dict([(k if k not in conflict_list else '_' + k, v) for k, v in output.items()])
    return result