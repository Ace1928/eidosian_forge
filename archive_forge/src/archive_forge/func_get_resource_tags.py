from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
def get_resource_tags():
    params = dict()
    if module.params.get('resource_id'):
        params['ResourceIds'] = module.params.get('resource_id')
    else:
        module.fail_json(msg='resource_id or resource_ids is required')
    if module.params.get('query') == 'health_check':
        params['ResourceType'] = 'healthcheck'
    else:
        params['ResourceType'] = 'hostedzone'
    return client.list_tags_for_resources(**params)