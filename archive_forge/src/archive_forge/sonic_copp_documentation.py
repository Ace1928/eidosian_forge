from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.copp.copp import CoppArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.config.copp.copp import Copp

    Main entry point for module execution

    :returns: the result form module invocation
    