from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.static_routes.static_routes import Static_routesArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.config.static_routes.static_routes import Static_routes

    Main entry point for module execution

    :returns: the result form module invocation
    