import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def wait_for_instance_state(module, client, instance_name, states):
    """
    `states` is a list of instance states that we are waiting for.
    """
    wait_timeout = module.params.get('wait_timeout')
    wait_max = time.time() + wait_timeout
    while wait_max > time.time():
        try:
            instance = find_instance_info(module, client, instance_name)
            if instance['state']['name'] in states:
                break
            time.sleep(5)
        except botocore.exceptions.ClientError as e:
            module.fail_json_aws(e)
    else:
        module.fail_json(msg=f'Timed out waiting for instance "{instance_name}" to get to one of the following states - {states}')