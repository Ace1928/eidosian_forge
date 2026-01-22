from __future__ import absolute_import, division, print_function
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import edit_config, to_request

    Main entry point for module execution

    :returns: the result form module invocation
    