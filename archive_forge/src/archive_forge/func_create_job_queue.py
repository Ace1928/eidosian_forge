from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.batch import set_api_params
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_job_queue(module, client):
    """
    Adds a Batch job queue

    :param module:
    :param client:
    :return:
    """
    changed = False
    params = ('job_queue_name', 'priority')
    api_params = set_api_params(module, params)
    if module.params['job_queue_state'] is not None:
        api_params['state'] = module.params['job_queue_state']
    api_params['computeEnvironmentOrder'] = get_compute_environment_order_list(module)
    try:
        if not module.check_mode:
            client.create_job_queue(**api_params)
        changed = True
    except (BotoCoreError, ClientError) as e:
        module.fail_json_aws(e, msg='Error creating compute environment')
    return changed