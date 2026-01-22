from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.l2_acls.l2_acls import L2_aclsArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.config.l2_acls.l2_acls import L2_acls

    Main entry point for module execution

    :returns: the result form module invocation
    