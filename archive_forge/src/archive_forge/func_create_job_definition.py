from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.batch import cc
from ansible_collections.amazon.aws.plugins.module_utils.batch import set_api_params
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_job_definition(module, batch_client):
    """
    Adds a Batch job definition

    :param module:
    :param batch_client:
    :return:
    """
    changed = False
    api_params = set_api_params(module, get_base_params())
    container_properties_params = set_api_params(module, get_container_property_params())
    retry_strategy_params = set_api_params(module, get_retry_strategy_params())
    api_params['retryStrategy'] = retry_strategy_params
    api_params['containerProperties'] = container_properties_params
    try:
        if not module.check_mode:
            batch_client.register_job_definition(**api_params)
        changed = True
    except (BotoCoreError, ClientError) as e:
        module.fail_json_aws(e, msg='Error registering job definition')
    return changed