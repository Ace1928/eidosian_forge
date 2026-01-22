from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.vyos import run_commands
def build_ping(dest, count, size=None, interval=None, source=None, ttl=None):
    cmd = 'ping {0} count {1}'.format(dest, str(count))
    if source:
        cmd += ' interface {0}'.format(source)
    if ttl:
        cmd += ' ttl {0}'.format(str(ttl))
    if size:
        cmd += ' size {0}'.format(str(size))
    if interval:
        cmd += ' interval {0}'.format(str(interval))
    return cmd