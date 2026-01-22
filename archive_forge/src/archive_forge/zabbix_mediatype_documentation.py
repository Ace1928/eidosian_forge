from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
Filters only the parameters that are different and need to be updated.

        Args:
            mediatype_id (int): ID of the mediatype to be updated.
            **kwargs: Parameters for the new mediatype.

        Returns:
            A tuple where the first element is a dictionary of parameters
            that need to be updated and the second one is a dictionary
            returned by diff() function with
            existing mediatype data and new params passed to it.
        