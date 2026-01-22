from time import sleep
from time import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def register_target(connection, module):
    """
    Registers a target to a target group

    :param module: ansible module object
    :param connection: boto3 connection
    :return:
    """
    target_az = module.params.get('target_az')
    target_group_arn = module.params.get('target_group_arn')
    target_id = module.params.get('target_id')
    target_port = module.params.get('target_port')
    target_status = module.params.get('target_status')
    target_status_timeout = module.params.get('target_status_timeout')
    changed = False
    if not target_group_arn:
        target_group_arn = convert_tg_name_to_arn(connection, module, module.params.get('target_group_name'))
    target = dict(Id=target_id)
    if target_az:
        target['AvailabilityZone'] = target_az
    if target_port:
        target['Port'] = target_port
    target_description = describe_targets(connection, module, target_group_arn, target)
    if 'Reason' in target_description['TargetHealth']:
        if target_description['TargetHealth']['Reason'] == 'Target.NotRegistered':
            try:
                register_target_with_backoff(connection, target_group_arn, target)
                changed = True
                if target_status:
                    target_status_check(connection, module, target_group_arn, target, target_status, target_status_timeout)
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                module.fail_json_aws(e, msg=f'Unable to deregister target {target}')
    target_descriptions = describe_targets(connection, module, target_group_arn)
    module.exit_json(changed=changed, target_health_descriptions=camel_dict_to_snake_dict(target_descriptions), target_group_arn=target_group_arn)