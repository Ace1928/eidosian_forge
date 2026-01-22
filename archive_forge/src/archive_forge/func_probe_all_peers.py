from __future__ import absolute_import, division, print_function
import re
import socket
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
def probe_all_peers(hosts, peers, myhostname):
    for host in hosts:
        host = host.strip()
        if host not in peers:
            probe(host, myhostname)