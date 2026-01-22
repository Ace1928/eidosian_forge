from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils import arguments, errors, utils
def build_api_payload(params):
    payload = arguments.get_mutation_payload(params, 'command', 'cron', 'handlers', 'high_flap_threshold', 'interval', 'low_flap_threshold', 'output_metric_format', 'output_metric_handlers', 'proxy_entity_name', 'publish', 'round_robin', 'runtime_assets', 'secrets', 'stdin', 'subscriptions', 'timeout', 'ttl')
    if params['proxy_requests']:
        payload['proxy_requests'] = arguments.get_spec_payload(params['proxy_requests'], 'entity_attributes', 'splay', 'splay_coverage')
    if params['check_hooks']:
        payload['check_hooks'] = utils.dict_to_single_item_dicts(params['check_hooks'])
    if params['env_vars']:
        payload['env_vars'] = utils.dict_to_key_value_strings(params['env_vars'])
    return payload