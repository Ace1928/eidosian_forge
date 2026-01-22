from time import sleep
from time import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
def update_broker(conn, module, broker_id):
    kwargs = _fill_kwargs(module, apply_defaults=False, ignore_create_params=True)
    wait = module.params.get('wait')
    broker_name = kwargs['BrokerName']
    del kwargs['BrokerName']
    kwargs['BrokerId'] = broker_id
    api_result = get_broker_info(conn, module, broker_id)
    if api_result['BrokerState'] != 'RUNNING':
        module.fail_json(msg=f'Cannot trigger update while broker ({broker_id}) is in state {api_result['BrokerState']}')
    if 'EngineVersion' in kwargs and kwargs['EngineVersion'] == 'latest':
        kwargs['EngineVersion'] = api_result['EngineVersion']
    result = {'broker_id': broker_id, 'broker_name': broker_name}
    changed = False
    if _needs_change(api_result, kwargs):
        changed = True
        if not module.check_mode:
            api_result = conn.update_broker(**kwargs)
    if wait:
        wait_for_status(conn, module)
    return {'broker': result, 'changed': changed}