from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.vyos import run_commands
def parse_rtt(rtt_info):
    rtt_re = re.compile('rtt (?:.*)=(?:\\s*)(?P<min>\\d*).(?:\\d*)/(?P<avg>\\d*).(?:\\d*)/(?P<max>\\d+).(?:\\d*)/(?P<mdev>\\d*)')
    rtt = rtt_re.match(rtt_info)
    return rtt.groupdict()