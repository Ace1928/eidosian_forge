from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.ucs.plugins.module_utils.ucs import UCSModule, ucs_argument_spec
def matches_existing_vnic_order(vnic, vnic_mo):
    if vnic['state'] == 'absent':
        kwargs = dict(admin_vcon='any')
        kwargs['order'] = 'unspecified'
    else:
        kwargs = dict(admin_vcon=vnic['admin_vcon'])
        kwargs['order'] = vnic['order']
    if vnic['transport'] == 'ethernet':
        kwargs['type'] = 'ether'
    else:
        kwargs['type'] = vnic['transport']
    return vnic_mo.check_prop_match(**kwargs)