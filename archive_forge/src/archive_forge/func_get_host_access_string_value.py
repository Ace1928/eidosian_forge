from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_host_access_string_value(self, host_dict):
    """
        Form host access string
        :host_dict Host access type info
        :return Host access data in string
        """
    if host_dict.get('host_id'):
        return self.get_host_obj(host_id=host_dict.get('host_id')).name + ','
    elif host_dict.get('host_name'):
        return host_dict.get('host_name') + ','
    elif host_dict.get('ip_address'):
        return host_dict.get('ip_address') + ','
    elif host_dict.get('subnet'):
        return host_dict.get('subnet') + ','
    elif host_dict.get('domain'):
        return '*.' + host_dict.get('domain') + ','
    elif host_dict.get('netgroup'):
        return '@' + host_dict.get('netgroup') + ','