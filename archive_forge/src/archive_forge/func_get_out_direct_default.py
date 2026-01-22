from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, get_nc_config, set_nc_config, check_ip_addr
def get_out_direct_default(out_direct):
    """get default out direct"""
    outdict = {'console': '1', 'monitor': '2', 'trapbuffer': '3', 'logbuffer': '4', 'snmp': '5', 'logfile': '6'}
    channel_id_default = outdict.get(out_direct)
    return channel_id_default