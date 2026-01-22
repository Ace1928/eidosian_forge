from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.urls import ConnectionError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def get_group_ids(rest_obj, module):
    """Getting the list of group ids filtered from the groups."""
    resp = rest_obj.get_all_report_details('GroupService/Groups')
    group_name = module.params.get('device_group_names')
    if resp['report_list']:
        grp_ids = [grp['Id'] for grp in resp['report_list'] for grpname in group_name if grp['Name'] == grpname]
        if len(set(group_name)) != len(set(grp_ids)):
            module.fail_json(msg="Unable to complete the operation because the entered target device group name(s) '{0}' are invalid.".format(','.join(set(group_name))))
    return grp_ids