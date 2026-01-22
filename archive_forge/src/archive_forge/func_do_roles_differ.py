from __future__ import absolute_import, division, print_function
from . import utils
def do_roles_differ(current, desired):
    if _do_rules_differ(current['rules'], desired['rules']):
        return True
    return utils.do_differ(current, desired, 'rules')