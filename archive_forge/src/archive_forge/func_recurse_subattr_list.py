from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import apply_diff_key, job_tracking
def recurse_subattr_list(subgroup, prefix, attr_detailed, attr_map, adv_list):
    if isinstance(subgroup, list):
        for each_sub in subgroup:
            nprfx = '{0}{1}{2}'.format(prefix, SEPRTR, each_sub.get('DisplayName'))
            if each_sub.get('SubAttributeGroups'):
                recurse_subattr_list(each_sub.get('SubAttributeGroups'), nprfx, attr_detailed, attr_map, adv_list)
            else:
                for attr in each_sub.get('Attributes'):
                    attr['prefix'] = nprfx
                    constr = '{0}{1}{2}'.format(nprfx, SEPRTR, attr['DisplayName'])
                    if constr in adv_list:
                        attr_detailed[constr] = attr['AttributeId']
                    attr_map[attr['AttributeId']] = attr