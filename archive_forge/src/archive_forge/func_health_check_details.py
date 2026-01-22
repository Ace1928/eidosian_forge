from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
def health_check_details():
    health_check_invocations = {'list': list_health_checks, 'details': get_health_check, 'status': get_health_check, 'failure_reason': get_health_check, 'count': get_count, 'tags': get_resource_tags}
    results = health_check_invocations[module.params.get('health_check_method')]()
    return results