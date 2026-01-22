from __future__ import (absolute_import, division, print_function)
import json
import socket
import copy
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
def get_ip_from_host(hostname):
    ipaddr = hostname
    try:
        result = socket.getaddrinfo(hostname, None)
        last_element = result[-1]
        ip_address = last_element[-1][0]
        if ip_address:
            ipaddr = ip_address
    except socket.gaierror:
        ipaddr = hostname
    except Exception:
        ipaddr = hostname
    return ipaddr