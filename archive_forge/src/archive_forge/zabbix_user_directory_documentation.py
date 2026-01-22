from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
 For future when < 6.4 disappears we should use this, now we cannot do this as at this point Zabbix version is unknown
    module = AnsibleModule(
        argument_spec=argument_spec,
        supports_check_mode=True,
        required_if=[
            ("state", "present", ("idp_type",)),
            ("idp_type", "ldap", ("host", "port", "base_dn", "search_attribute"), False),
            ("idp_type", "saml", ("idp_entityid", "sp_entityid", "sso_url", "username_attribute"), False),
            ("provision_status", "true", ("provision_groups"))
        ]
    )
    