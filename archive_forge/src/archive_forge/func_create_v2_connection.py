from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, human_to_bytes
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def create_v2_connection(module, blade):
    """Create connection between REST 2 capable arrays"""
    changed = True
    if blade.get_array_connections().total_item_count == FAN_OUT_MAXIMUM:
        module.fail_json(msg='FlashBlade fan-out maximum of {0} already reached'.format(FAN_OUT_MAXIMUM))
    try:
        remote_system = flashblade.Client(target=module.params['target_url'], api_token=module.params['target_api'])
    except Exception:
        module.fail_json(msg='Failed to connect to remote array {0}.'.format(module.params['target_url']))
    remote_array = list(remote_system.get_arrays().items)[0].name
    remote_conn_cnt = remote_system.get_array_connections().total_item_count
    if remote_conn_cnt == FAN_IN_MAXIMUM:
        module.fail_json(msg='Remote array {0} already connected to {1} other array. Fan-In not supported'.format(remote_array, remote_conn_cnt))
    connection_key = list(remote_system.post_array_connections_connection_key().items)[0].connection_key
    if module.params['default_limit'] or module.params['window_limit']:
        if THROTTLE_API_VERSION in list(blade.get_versions().items):
            if THROTTLE_API_VERSION not in list(remote_system.get_versions().items):
                module.fail_json(msg='Remote array does not support throttling')
            if module.params['window_limit']:
                if not module.params['window_start']:
                    module.params['window_start'] = '12AM'
                if not module.params['window_end']:
                    module.params['window_end'] = '12AM'
                window = flashblade.TimeWindow(start=_convert_to_millisecs(module.params['window_start']), end=_convert_to_millisecs(module.params['window_end']))
            if module.params['window_limit'] and module.params['default_limit']:
                throttle = flashblade.Throttle(default_limit=human_to_bytes(module.params['default_limit']), window_limit=human_to_bytes(module.params['window_limit']), window=window)
            elif module.params['window_limit'] and (not module.params['default_limit']):
                throttle = flashblade.Throttle(window_limit=human_to_bytes(module.params['window_limit']), window=window)
            else:
                throttle = flashblade.Throttle(default_limit=human_to_bytes(module.params['default_limit']))
            connection_info = ArrayConnectionPost(management_address=module.params['target_url'], replication_addresses=module.params['target_repl'], encrypted=module.params['encrypted'], connection_key=connection_key, throttle=throttle)
        else:
            connection_info = ArrayConnectionPost(management_address=module.params['target_url'], replication_addresses=module.params['target_repl'], encrypted=module.params['encrypted'], connection_key=connection_key)
    else:
        connection_info = ArrayConnectionPost(management_address=module.params['target_url'], replication_addresses=module.params['target_repl'], encrypted=module.params['encrypted'], connection_key=connection_key)
    if not module.check_mode:
        res = blade.post_array_connections(array_connection=connection_info)
        if res.status_code != 200:
            module.fail_json(msg='Failed to connect to remote array {0}. Error: {1}'.format(remote_array, res.errors[0].message))
    module.exit_json(changed=changed)