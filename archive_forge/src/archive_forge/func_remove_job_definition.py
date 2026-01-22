from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.batch import cc
from ansible_collections.amazon.aws.plugins.module_utils.batch import set_api_params
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def remove_job_definition(module, batch_client):
    """
    Remove a Batch job definition

    :param module:
    :param batch_client:
    :return:
    """
    changed = False
    try:
        if not module.check_mode:
            batch_client.deregister_job_definition(jobDefinition=module.params['job_definition_arn'])
        changed = True
    except (BotoCoreError, ClientError) as e:
        module.fail_json_aws(e, msg='Error removing job definition')
    return changed