from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def list_target_groups():
    load_balancer_arn = module.params.get('load_balancer_arn')
    target_group_arns = module.params.get('target_group_arns')
    names = module.params.get('names')
    collect_targets_health = module.params.get('collect_targets_health')
    try:
        if not load_balancer_arn and (not target_group_arns) and (not names):
            target_groups = get_paginator()
        if load_balancer_arn:
            target_groups = get_paginator(LoadBalancerArn=load_balancer_arn)
        if target_group_arns:
            target_groups = get_paginator(TargetGroupArns=target_group_arns)
        if names:
            target_groups = get_paginator(Names=names)
    except is_boto3_error_code('TargetGroupNotFound'):
        module.exit_json(target_groups=[])
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to list target groups')
    for target_group in target_groups['TargetGroups']:
        target_group.update(get_target_group_attributes(target_group['TargetGroupArn']))
    snaked_target_groups = [camel_dict_to_snake_dict(target_group) for target_group in target_groups['TargetGroups']]
    for snaked_target_group in snaked_target_groups:
        snaked_target_group['tags'] = get_target_group_tags(snaked_target_group['target_group_arn'])
        if collect_targets_health:
            snaked_target_group['targets_health_description'] = [camel_dict_to_snake_dict(target) for target in get_target_group_targets_health(snaked_target_group['target_group_arn'])]
    module.exit_json(target_groups=snaked_target_groups)