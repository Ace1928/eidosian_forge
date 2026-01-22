from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_or_update_fargate_profile(client, module):
    name = module.params.get('name')
    subnets = module.params['subnets']
    role_arn = module.params['role_arn']
    cluster_name = module.params['cluster_name']
    selectors = module.params['selectors']
    tags = module.params['tags'] or {}
    wait = module.params.get('wait')
    fargate_profile = get_fargate_profile(client, module, name, cluster_name)
    if fargate_profile:
        changed = False
        if set(fargate_profile['podExecutionRoleArn']) != set(role_arn):
            module.fail_json(msg='Cannot modify Execution Role')
        if set(fargate_profile['subnets']) != set(subnets):
            module.fail_json(msg='Cannot modify Subnets')
        if fargate_profile['selectors'] != selectors:
            module.fail_json(msg='Cannot modify Selectors')
        changed = validate_tags(client, module, fargate_profile)
        if wait:
            wait_until(client, module, 'fargate_profile_active', name, cluster_name)
        fargate_profile = get_fargate_profile(client, module, name, cluster_name)
        module.exit_json(changed=changed, **camel_dict_to_snake_dict(fargate_profile))
    if module.check_mode:
        module.exit_json(changed=True)
    check_profiles_status(client, module, cluster_name)
    try:
        params = dict(fargateProfileName=name, podExecutionRoleArn=role_arn, subnets=subnets, clusterName=cluster_name, selectors=selectors, tags=tags)
        fargate_profile = client.create_fargate_profile(**params)
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        module.fail_json_aws(e, msg=f"Couldn't create fargate profile {name}")
    if wait:
        wait_until(client, module, 'fargate_profile_active', name, cluster_name)
    fargate_profile = get_fargate_profile(client, module, name, cluster_name)
    module.exit_json(changed=True, **camel_dict_to_snake_dict(fargate_profile))