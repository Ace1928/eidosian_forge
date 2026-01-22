from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def is_nfs_have_host_with_host_string(nfs_details):
    """ Check whether nfs host is already added using host by string method

    :param nfs_details: nfs details
    :return: True if nfs have host already added with host string method else False
    :rtype: bool
    """
    host_obj_params = ('no_access_hosts_string', 'read_only_hosts_string', 'read_only_root_hosts_string', 'read_write_hosts_string', 'read_write_root_hosts_string')
    for host_obj_param in host_obj_params:
        if nfs_details.get(host_obj_param):
            return True
    return False