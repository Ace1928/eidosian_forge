from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils._text import to_native
@staticmethod
def get_api_net_stack_instance(tcpip_stack):
    """Get TCP/IP stack instance name or key"""
    net_stack_instance = None
    if tcpip_stack == 'default':
        net_stack_instance = 'defaultTcpipStack'
    elif tcpip_stack == 'provisioning':
        net_stack_instance = 'vSphereProvisioning'
    elif tcpip_stack == 'vmotion':
        net_stack_instance = 'vmotion'
    elif tcpip_stack == 'vxlan':
        net_stack_instance = 'vxlan'
    elif tcpip_stack == 'defaultTcpipStack':
        net_stack_instance = 'default'
    elif tcpip_stack == 'vSphereProvisioning':
        net_stack_instance = 'provisioning'
    return net_stack_instance