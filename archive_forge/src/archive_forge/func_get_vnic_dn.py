from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.ucs.plugins.module_utils.ucs import UCSModule, ucs_argument_spec
def get_vnic_dn(sp_dn, transport, name):
    if transport == 'ethernet':
        return sp_dn + '/ether-' + name
    return sp_dn + '/fc-' + name