from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
from datetime import datetime
def generate_default_dict(module, blade):
    default_info = {}
    defaults = blade.arrays.list_arrays().items[0]
    default_info['flashblade_name'] = defaults.name
    default_info['purity_version'] = defaults.version
    default_info['filesystems'] = len(blade.file_systems.list_file_systems().items)
    default_info['snapshots'] = len(blade.file_system_snapshots.list_file_system_snapshots().items)
    default_info['buckets'] = len(blade.buckets.list_buckets().items)
    default_info['object_store_users'] = len(blade.object_store_users.list_object_store_users().items)
    default_info['object_store_accounts'] = len(blade.object_store_accounts.list_object_store_accounts().items)
    default_info['blades'] = len(blade.blade.list_blades().items)
    default_info['certificates'] = len(blade.certificates.list_certificates().items)
    default_info['total_capacity'] = blade.arrays.list_arrays_space().items[0].capacity
    api_version = blade.api_version.list_versions().versions
    default_info['api_versions'] = api_version
    if POLICIES_API_VERSION in api_version:
        default_info['policies'] = len(blade.policies.list_policies().items)
    if CERT_GROUPS_API_VERSION in api_version:
        default_info['certificate_groups'] = len(blade.certificate_groups.list_certificate_groups().items)
    if REPLICATION_API_VERSION in api_version:
        default_info['fs_replicas'] = len(blade.file_system_replica_links.list_file_system_replica_links().items)
        default_info['remote_credentials'] = len(blade.object_store_remote_credentials.list_object_store_remote_credentials().items)
        default_info['bucket_replicas'] = len(blade.bucket_replica_links.list_bucket_replica_links().items)
        default_info['connected_arrays'] = len(blade.array_connections.list_array_connections().items)
        default_info['targets'] = len(blade.targets.list_targets().items)
        default_info['kerberos_keytabs'] = len(blade.keytabs.list_keytabs().items)
    if MIN_32_API in api_version:
        blade = get_system(module)
        blade_info = list(blade.get_arrays().items)[0]
        default_info['object_store_virtual_hosts'] = len(blade.get_object_store_virtual_hosts().items)
        default_info['api_clients'] = len(blade.get_api_clients().items)
        default_info['idle_timeout'] = int(blade_info.idle_timeout / 60000)
        if list(blade.get_arrays_eula().items)[0].signature.accepted:
            default_info['EULA'] = 'Signed'
        else:
            default_info['EULA'] = 'Not Signed'
        if NFS_POLICY_API_VERSION in api_version:
            admin_settings = list(blade.get_admins_settings().items)[0]
            default_info['max_login_attempts'] = admin_settings.max_login_attempts
            default_info['min_password_length'] = admin_settings.min_password_length
            if admin_settings.lockout_duration:
                default_info['lockout_duration'] = str(admin_settings.lockout_duration / 1000) + ' seconds'
        if NFS_POLICY_API_VERSION in api_version:
            default_info['smb_mode'] = blade_info.smb_mode
        if VSO_VERSION in api_version:
            default_info['timezone'] = blade_info.time_zone
        if DRIVES_API_VERSION in api_version:
            default_info['product_type'] = getattr(blade_info, 'product_type', 'Unknown')
        if SECURITY_API_VERSION in api_version:
            dar = blade_info.encryption.data_at_rest
            default_info['encryption'] = {'data_at_rest_enabled': dar.enabled, 'data_at_rest_algorithms': dar.algorithms, 'data_at_rest_entropy_source': dar.entropy_source}
            keys = list(blade.get_support_verification_keys().items)
            default_info['support_keys'] = {}
            for key in range(0, len(keys)):
                keyname = keys[key].name
                default_info['support_keys'][keyname] = {keys[key].verification_key}
            default_info['security_update'] = getattr(blade_info, 'security_update', None)
    return default_info