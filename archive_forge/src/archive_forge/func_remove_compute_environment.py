import re
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.arn import validate_aws_arn
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def remove_compute_environment(module, client):
    """
    Remove a Batch compute environment

    :param module:
    :param client:
    :return:
    """
    changed = False
    api_params = {'computeEnvironment': module.params['compute_environment_name']}
    try:
        if not module.check_mode:
            client.delete_compute_environment(**api_params)
        changed = True
    except (ClientError, BotoCoreError) as e:
        module.fail_json_aws(e, msg='Error removing compute environment')
    return changed