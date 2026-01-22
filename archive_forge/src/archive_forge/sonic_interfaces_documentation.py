from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.interfaces.interfaces import InterfacesArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.config.interfaces.interfaces import Interfaces

    Main entry point for module execution

    :returns: the result form module invocation
    