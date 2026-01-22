from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def delete_fargate_profile(client, module):
    name = module.params.get('name')
    cluster_name = module.params['cluster_name']
    existing = get_fargate_profile(client, module, name, cluster_name)
    wait = module.params.get('wait')
    if not existing or existing['status'] == 'DELETING':
        module.exit_json(changed=False)
    if not module.check_mode:
        check_profiles_status(client, module, cluster_name)
        try:
            client.delete_fargate_profile(clusterName=cluster_name, fargateProfileName=name)
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            module.fail_json_aws(e, msg=f"Couldn't delete fargate profile {name}")
        if wait:
            wait_until(client, module, 'fargate_profile_deleted', name, cluster_name)
    module.exit_json(changed=True)