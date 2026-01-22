from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.ucs.plugins.module_utils.ucs import UCSModule, ucs_argument_spec
def check_iscsi_vnic_props(ucs, module, dn):
    props_match = True
    if module.params.get('iscsi_vnic_list'):
        for iscsi_vnic in module.params['iscsi_vnic_list']:
            child_dn = dn + '/iscsi-' + iscsi_vnic['name']
            mo_1 = ucs.login_handle.query_dn(child_dn)
            if mo_1:
                if iscsi_vnic['state'] == 'absent':
                    props_match = False
                    break
                else:
                    kwargs = dict(vnic_name=iscsi_vnic['overlay_vnic'])
                    kwargs['adaptor_profile_name'] = iscsi_vnic['iscsi_adapter_policy']
                    kwargs['addr'] = iscsi_vnic['mac_address']
                    if mo_1.check_prop_match(**kwargs):
                        child_dn = child_dn + '/vlan'
                        mo_2 = ucs.login_handle.query_dn(child_dn)
                        if mo_2:
                            kwargs = dict(vlan_name=iscsi_vnic['vlan_name'])
                            if not mo_2.check_prop_match(**kwargs):
                                props_match = False
                                break
                    else:
                        props_match = False
                        break
            elif iscsi_vnic['state'] == 'present':
                props_match = False
                break
    return props_match