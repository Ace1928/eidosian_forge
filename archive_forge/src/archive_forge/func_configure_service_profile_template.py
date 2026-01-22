from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.ucs.plugins.module_utils.ucs import UCSModule, ucs_argument_spec
def configure_service_profile_template(ucs, module):
    from ucsmsdk.mometa.ls.LsServer import LsServer
    from ucsmsdk.mometa.vnic.VnicConnDef import VnicConnDef
    from ucsmsdk.mometa.vnic.VnicIScsiNode import VnicIScsiNode
    from ucsmsdk.mometa.ls.LsRequirement import LsRequirement
    from ucsmsdk.mometa.ls.LsPower import LsPower
    from ucsmsdk.mometa.lstorage.LstorageProfileBinding import LstorageProfileBinding
    from ucsmsdk.mometa.mgmt.MgmtInterface import MgmtInterface
    from ucsmsdk.mometa.mgmt.MgmtVnet import MgmtVnet
    from ucsmsdk.mometa.vnic.VnicIpV4MgmtPooledAddr import VnicIpV4MgmtPooledAddr
    if not module.check_mode:
        try:
            mo = LsServer(parent_mo_or_dn=module.params['org_dn'], bios_profile_name=module.params['bios_policy'], boot_policy_name=module.params['boot_policy'], descr=module.params['description'], ext_ip_state=module.params['mgmt_ip_state'], ext_ip_pool_name=module.params['mgmt_ip_pool'], host_fw_policy_name=module.params['host_firmware_package'], ident_pool_name=module.params['uuid_pool'], kvm_mgmt_policy_name=module.params['kvm_mgmt_policy'], local_disk_policy_name=module.params['local_disk_policy'], maint_policy_name=module.params['maintenance_policy'], mgmt_access_policy_name=module.params['ipmi_access_profile'], name=module.params['name'], power_policy_name=module.params['power_control_policy'], power_sync_policy_name=module.params['power_sync_policy'], scrub_policy_name=module.params['scrub_policy'], sol_policy_name=module.params['sol_policy'], stats_policy_name=module.params['threshold_policy'], type=module.params['template_type'], usr_lbl=module.params['user_label'], vmedia_policy_name=module.params['vmedia_policy'])
            if module.params['storage_profile']:
                mo_1 = LstorageProfileBinding(parent_mo_or_dn=mo, storage_profile_name=module.params['storage_profile'])
            if module.params['mgmt_interface_mode']:
                mo_1 = MgmtInterface(parent_mo_or_dn=mo, mode=module.params['mgmt_interface_mode'], ip_v4_state='pooled')
                mo_2 = MgmtVnet(parent_mo_or_dn=mo_1, id='1', name=module.params['mgmt_vnet_name'])
                VnicIpV4MgmtPooledAddr(parent_mo_or_dn=mo_2, name=module.params['mgmt_inband_pool_name'])
            mo_1 = VnicConnDef(parent_mo_or_dn=mo, lan_conn_policy_name=module.params['lan_connectivity_policy'], san_conn_policy_name=module.params['san_connectivity_policy'])
            if module.params['iqn_pool']:
                mo_1 = VnicIScsiNode(parent_mo_or_dn=mo, iqn_ident_pool_name=module.params['iqn_pool'])
            admin_state = 'admin-' + module.params['power_state']
            mo_1 = LsPower(parent_mo_or_dn=mo, state=admin_state)
            if module.params['server_pool']:
                mo_1 = LsRequirement(parent_mo_or_dn=mo, name=module.params['server_pool'], qualifier=module.params['server_pool_qualification'])
            ucs.login_handle.add_mo(mo, True)
            ucs.login_handle.commit()
        except Exception as e:
            ucs.result['msg'] = 'setup error: %s ' % str(e)
            module.fail_json(**ucs.result)
    ucs.result['changed'] = True