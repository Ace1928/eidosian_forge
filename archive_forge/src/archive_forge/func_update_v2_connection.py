from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, human_to_bytes
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def update_v2_connection(module, blade):
    """Update REST 2 based array connection"""
    changed = False
    versions = list(blade.get_versions().items)
    remote_blade = flashblade.Client(target=module.params['target_url'], api_token=module.params['target_api'])
    remote_name = list(remote_blade.get_arrays().items)[0].name
    remote_connection = list(blade.get_array_connections(filter="remote.name='" + remote_name + "'").items)[0]
    if remote_connection.management_address is None:
        module.fail_json(msg='Update can only happen from the array that formed the connection')
    if module.params['encrypted'] != remote_connection.encrypted:
        if module.params['encrypted'] and blade.get_file_system_replica_links().total_item_count != 0:
            module.fail_json(msg='Cannot turn array connection encryption on if file system replica links exist')
    current_connection = {'encrypted': remote_connection.encrypted, 'replication_addresses': sorted(remote_connection.replication_addresses), 'throttle': []}
    if not remote_connection.throttle.default_limit and (not remote_connection.throttle.window_limit):
        if (module.params['default_limit'] or module.params['window_limit']) and blade.get_bucket_replica_links().total_item_count != 0:
            module.fail_json(msg='Cannot set throttle when bucket replica links already exist')
    if THROTTLE_API_VERSION in versions:
        current_connection['throttle'] = {'default_limit': remote_connection.throttle.default_limit, 'window_limit': remote_connection.throttle.window_limit, 'start': remote_connection.throttle.window.start, 'end': remote_connection.throttle.window.end}
    if module.params['encrypted']:
        encryption = module.params['encrypted']
    else:
        encryption = remote_connection.encrypted
    if module.params['target_repl']:
        target_repl = sorted(module.params['target_repl'])
    else:
        target_repl = remote_connection.replication_addresses
    if module.params['default_limit']:
        default_limit = human_to_bytes(module.params['default_limit'])
        if default_limit == 0:
            default_limit = None
    else:
        default_limit = remote_connection.throttle.default_limit
    if module.params['window_limit']:
        window_limit = human_to_bytes(module.params['window_limit'])
    else:
        window_limit = remote_connection.throttle.window_limit
    if module.params['window_start']:
        start = _convert_to_millisecs(module.params['window_start'])
    else:
        start = remote_connection.throttle.window.start
    if module.params['window_end']:
        end = _convert_to_millisecs(module.params['window_end'])
    else:
        end = remote_connection.throttle.window.end
    new_connection = {'encrypted': encryption, 'replication_addresses': target_repl, 'throttle': []}
    if THROTTLE_API_VERSION in versions:
        new_connection['throttle'] = {'default_limit': default_limit, 'window_limit': window_limit, 'start': start, 'end': end}
    if new_connection != current_connection:
        changed = True
        if not module.check_mode:
            if THROTTLE_API_VERSION in versions:
                window = flashblade.TimeWindow(start=new_connection['throttle']['start'], end=new_connection['throttle']['end'])
                throttle = flashblade.Throttle(default_limit=new_connection['throttle']['default_limit'], window_limit=new_connection['throttle']['window_limit'], window=window)
                connection_info = ArrayConnectionPost(replication_addresses=new_connection['replication_addresses'], encrypted=new_connection['encrypted'], throttle=throttle)
            else:
                connection_info = ArrayConnection(replication_addresses=new_connection['replication_addresses'], encrypted=new_connection['encrypted'])
            res = blade.patch_array_connections(remote_names=[remote_name], array_connection=connection_info)
            if res.status_code != 200:
                module.fail_json(msg='Failed to update connection to remote array {0}. Error: {1}'.format(remote_name, res.errors[0].message))
    module.exit_json(changed=changed)