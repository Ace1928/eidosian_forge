from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import run_commands
def get_rtt(results_list, packet_loss, location):
    rtt = dict(min=None, avg=None, max=None)
    if packet_loss != '100.00%':
        rtt_string = results_list[location]
        base = rtt_string.split('=')[1]
        rtt_list = base.split('/')
        rtt['min'] = float(rtt_list[0].lstrip())
        rtt['avg'] = float(rtt_list[1])
        rtt['max'] = float(rtt_list[2][:-3])
    return rtt