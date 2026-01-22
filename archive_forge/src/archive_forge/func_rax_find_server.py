from __future__ import (absolute_import, division, print_function)
import os
import re
from uuid import UUID
from ansible.module_utils.six import text_type, binary_type
def rax_find_server(module, rax_module, server):
    """Find a Cloud Server by ID or name"""
    cs = rax_module.cloudservers
    try:
        UUID(server)
        server = cs.servers.get(server)
    except ValueError:
        servers = cs.servers.list(search_opts=dict(name='^%s$' % server))
        if not servers:
            module.fail_json(msg='No Server was matched by name, try using the Server ID instead')
        if len(servers) > 1:
            module.fail_json(msg='Multiple servers matched by name, try using the Server ID instead')
        server = servers[0]
    return server