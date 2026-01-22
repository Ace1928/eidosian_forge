from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def convert_subscr_params(params_dict):
    """Converts the passed params dictionary to string.

    Args:
        params_dict (list): Dictionary which needs to be converted.

    Returns:
        Parameters string.
    """
    params_list = []
    for param, val in iteritems(params_dict):
        if val is False:
            val = 'false'
        elif val is True:
            val = 'true'
        params_list.append('%s = %s' % (param, val))
    return ', '.join(params_list)