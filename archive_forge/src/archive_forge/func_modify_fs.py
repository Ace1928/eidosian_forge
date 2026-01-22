from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, human_to_bytes
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def modify_fs(module, blade):
    """Modify Filesystem"""
    changed = False
    change_export = False
    change_share = False
    change_ca = False
    mod_fs = False
    attr = {}
    if module.params['policy'] and module.params['policy_state'] == 'present':
        try:
            policy = blade.policies.list_policy_filesystems(policy_names=[module.params['policy']], member_names=[module.params['name']])
        except Exception:
            module.fail_json(msg='Policy {0} does not exist.'.format(module.params['policy']))
        if not policy.items:
            try:
                blade.policies.create_policy_filesystems(policy_names=[module.params['policy']], member_names=[module.params['name']])
                mod_fs = True
            except Exception:
                module.fail_json(msg='Failed to add filesystem {0} to policy {1}.'.format(module.params['name'], module.params['polict']))
    if module.params['policy'] and module.params['policy_state'] == 'absent':
        try:
            policy = blade.policies.list_policy_filesystems(policy_names=[module.params['policy']], member_names=[module.params['name']])
        except Exception:
            module.fail_json(msg='Policy {0} does not exist.'.format(module.params['policy']))
        if len(policy.items) == 1:
            try:
                blade.policies.delete_policy_filesystems(policy_names=[module.params['policy']], member_names=[module.params['name']])
                mod_fs = True
            except Exception:
                module.fail_json(msg='Failed to remove filesystem {0} to policy {1}.'.format(module.params['name'], module.params['polict']))
    if module.params['user_quota']:
        user_quota = human_to_bytes(module.params['user_quota'])
    if module.params['group_quota']:
        group_quota = human_to_bytes(module.params['group_quota'])
    fsys = get_fs(module, blade)
    if fsys.destroyed:
        attr['destroyed'] = False
        mod_fs = True
    if module.params['size']:
        if human_to_bytes(module.params['size']) != fsys.provisioned:
            attr['provisioned'] = human_to_bytes(module.params['size'])
            mod_fs = True
    api_version = blade.api_version.list_versions().versions
    if NFSV4_API_VERSION in api_version:
        v3_state = v4_state = None
        if module.params['nfsv3'] and (not fsys.nfs.v3_enabled):
            v3_state = module.params['nfsv3']
        if not module.params['nfsv3'] and fsys.nfs.v3_enabled:
            v3_state = module.params['nfsv3']
        if module.params['nfsv4'] and (not fsys.nfs.v4_1_enabled):
            v4_state = module.params['nfsv4']
        if not module.params['nfsv4'] and fsys.nfs.v4_1_enabled:
            v4_state = module.params['nfsv4']
        if v3_state is not None or v4_state is not None:
            attr['nfs'] = NfsRule(v4_1_enabled=v4_state, v3_enabled=v3_state)
            mod_fs = True
        if module.params['nfsv3'] or (module.params['nfsv4'] and fsys.nfs.v3_enabled) or fsys.nfs.v4_1_enabled:
            if module.params['nfs_rules'] is not None:
                if fsys.nfs.rules != module.params['nfs_rules']:
                    attr['nfs'] = NfsRule(rules=module.params['nfs_rules'])
                    mod_fs = True
        if module.params['user_quota'] and user_quota != fsys.default_user_quota:
            attr['default_user_quota'] = user_quota
            mod_fs = True
        if module.params['group_quota'] and group_quota != fsys.default_group_quota:
            attr['default_group_quota'] = group_quota
            mod_fs = True
    else:
        if module.params['nfsv3'] and (not fsys.nfs.enabled):
            attr['nfs'] = NfsRule(enabled=module.params['nfsv3'])
            mod_fs = True
        if not module.params['nfsv3'] and fsys.nfs.enabled:
            attr['nfs'] = NfsRule(enabled=module.params['nfsv3'])
            mod_fs = True
        if module.params['nfsv3'] and fsys.nfs.enabled:
            if fsys.nfs.rules != module.params['nfs_rules']:
                attr['nfs'] = NfsRule(rules=module.params['nfs_rules'])
                mod_fs = True
    if REPLICATION_API_VERSION in api_version:
        if module.params['smb'] and (not fsys.smb.enabled):
            if MULTIPROTOCOL_API_VERSION in api_version:
                attr['smb'] = SmbRule(enabled=module.params['smb'])
            else:
                attr['smb'] = SmbRule(enabled=module.params['smb'], acl_mode=module.params['smb_aclmode'])
            mod_fs = True
        if not module.params['smb'] and fsys.smb.enabled:
            attr['smb'] = ProtocolRule(enabled=module.params['smb'])
            mod_fs = True
        if module.params['smb'] and fsys.smb.enabled and (MULTIPROTOCOL_API_VERSION not in api_version):
            if fsys.smb.acl_mode != module.params['smb_aclmode']:
                attr['smb'] = SmbRule(enabled=module.params['smb'], acl_mode=module.params['smb_aclmode'])
                mod_fs = True
    else:
        if module.params['smb'] and (not fsys.smb.enabled):
            attr['smb'] = ProtocolRule(enabled=module.params['smb'])
            mod_fs = True
        if not module.params['smb'] and fsys.smb.enabled:
            attr['smb'] = ProtocolRule(enabled=module.params['smb'])
            mod_fs = True
    if module.params['http'] and (not fsys.http.enabled):
        attr['http'] = ProtocolRule(enabled=module.params['http'])
        mod_fs = True
    if not module.params['http'] and fsys.http.enabled:
        attr['http'] = ProtocolRule(enabled=module.params['http'])
        mod_fs = True
    if module.params['snapshot'] and (not fsys.snapshot_directory_enabled):
        attr['snapshot_directory_enabled'] = module.params['snapshot']
        mod_fs = True
    if not module.params['snapshot'] and fsys.snapshot_directory_enabled:
        attr['snapshot_directory_enabled'] = module.params['snapshot']
        mod_fs = True
    if module.params['fastremove'] and (not fsys.fast_remove_directory_enabled):
        attr['fast_remove_directory_enabled'] = module.params['fastremove']
        mod_fs = True
    if not module.params['fastremove'] and fsys.fast_remove_directory_enabled:
        attr['fast_remove_directory_enabled'] = module.params['fastremove']
        mod_fs = True
    if HARD_LIMIT_API_VERSION in api_version:
        if not module.params['hard_limit'] and fsys.hard_limit_enabled:
            attr['hard_limit_enabled'] = module.params['hard_limit']
            mod_fs = True
        if module.params['hard_limit'] and (not fsys.hard_limit_enabled):
            attr['hard_limit_enabled'] = module.params['hard_limit']
            mod_fs = True
    if MULTIPROTOCOL_API_VERSION in api_version:
        if module.params['safeguard_acls'] and (not fsys.multi_protocol.safeguard_acls):
            attr['multi_protocol'] = MultiProtocolRule(safeguard_acls=True)
            mod_fs = True
        if not module.params['safeguard_acls'] and fsys.multi_protocol.safeguard_acls:
            attr['multi_protocol'] = MultiProtocolRule(safeguard_acls=False)
            mod_fs = True
        if module.params['access_control'] != fsys.multi_protocol.access_control_style:
            attr['multi_protocol'] = MultiProtocolRule(access_control_style=module.params['access_control'])
            mod_fs = True
    if REPLICATION_API_VERSION in api_version:
        if module.params['writable'] is not None:
            if not module.params['writable'] and fsys.writable:
                attr['writable'] = module.params['writable']
                mod_fs = True
            if module.params['writable'] and (not fsys.writable) and (fsys.promotion_status == 'promoted'):
                attr['writable'] = module.params['writable']
                mod_fs = True
        if module.params['promote'] is not None:
            if module.params['promote'] and fsys.promotion_status != 'promoted':
                attr['requested_promotion_state'] = 'promoted'
                mod_fs = True
            if not module.params['promote'] and fsys.promotion_status == 'promoted':
                try:
                    blade.file_system_replica_links.list_file_system_replica_links(local_file_system_names=[module.params['name']]).items[0]
                except Exception:
                    module.fail_json(msg='Filesystem {0} not demoted. Not in a replica-link'.format(module.params['name']))
                attr['requested_promotion_state'] = 'demoted'
                mod_fs = True
    if mod_fs:
        changed = True
        if not module.check_mode:
            n_attr = FileSystem(**attr)
            if REPLICATION_API_VERSION in api_version:
                try:
                    blade.file_systems.update_file_systems(name=module.params['name'], attributes=n_attr, discard_non_snapshotted_data=module.params['discard_snaps'])
                except rest.ApiException as err:
                    message = json.loads(err.body)['errors'][0]['message']
                    module.fail_json(msg='Failed to update filesystem {0}. Error {1}'.format(module.params['name'], message))
            else:
                try:
                    blade.file_systems.update_file_systems(name=module.params['name'], attributes=n_attr)
                except rest.ApiException as err:
                    message = json.loads(err.body)['errors'][0]['message']
                    module.fail_json(msg='Failed to update filesystem {0}. Error {1}'.format(module.params['name'], message))
    system = get_system(module)
    current_fs = list(system.get_file_systems(filter="name='" + module.params['name'] + "'").items)[0]
    if EXPORT_POLICY_API_VERSION in api_version and module.params['export_policy']:
        change_export = False
        if current_fs.nfs.export_policy.name and current_fs.nfs.export_policy.name != module.params['export_policy']:
            change_export = True
        if not current_fs.nfs.export_policy.name and module.params['export_policy']:
            change_export = True
        if change_export and (not module.check_mode):
            export_attr = FileSystemPatch(nfs=NfsPatch(export_policy=Reference(name=module.params['export_policy'])))
            res = system.patch_file_systems(names=[module.params['name']], file_system=export_attr)
            if res.status_code != 200:
                module.fail_json(msg='Failed to modify export policy {1} for filesystem {0}. Error: {2}'.format(module.params['name'], module.params['export_policy'], res.errors[0].message))
    if SMB_POLICY_API_VERSION in api_version and module.params['client_policy']:
        change_client = False
        if current_fs.smb.client_policy.name and current_fs.smb.client_policy.name != module.params['client_policy']:
            change_client = True
        if not current_fs.smb.client_policy.name and module.params['client_policy']:
            change_client = True
        if change_client and (not module.check_mode):
            client_attr = FileSystemPatch(smb=Smb(client_policy=Reference(name=module.params['client_policy'])))
            res = system.patch_file_systems(names=[module.params['name']], file_system=client_attr)
            if res.status_code != 200:
                module.fail_json(msg='Failed to modify client policy {1} for filesystem {0}. Error: {2}'.format(module.params['name'], module.params['client_policy'], res.errors[0].message))
    if SMB_POLICY_API_VERSION in api_version and module.params['share_policy']:
        change_share = False
        if current_fs.smb.share_policy.name and current_fs.smb.share_policy.name != module.params['share_policy']:
            change_share = True
        if not current_fs.smb.share_policy.name and module.params['share_policy']:
            change_share = True
        if change_share and (not module.check_mode):
            share_attr = FileSystemPatch(smb=Smb(share_policy=Reference(name=module.params['share_policy'])))
            res = system.patch_file_systems(names=[module.params['name']], file_system=share_attr)
            if res.status_code != 200:
                module.fail_json(msg='Failed to modify share policy {1} for filesystem {0}. Error: {2}'.format(module.params['name'], module.params['share_policy'], res.errors[0].message))
    if CA_API_VERSION in api_version:
        change_ca = False
        if module.params['continuous_availability'] != current_fs.continuous_availability_enabled:
            change_ca = True
            if not module.check_mode:
                ca_attr = FileSystemPatch(smb=Smb(continuous_availability_enabled=module.params['continuous_availability']))
                res = system.patch_file_systems(names=[module.params['name']], file_system=ca_attr)
                if res.status_code != 200:
                    module.fail_json(msg='Failed to modify continuous availability for filesystem {0}. Error: {1}'.format(module.params['name'], res.errors[0].message))
    module.exit_json(changed=changed or change_export or change_share or change_ca)