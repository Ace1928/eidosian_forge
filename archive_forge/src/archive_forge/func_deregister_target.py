from time import sleep
from time import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def deregister_target(connection, module):
    """
    Deregisters a target to a target group

    :param module: ansible module object
    :param connection: boto3 connection
    :return:
    """
    deregister_unused = module.params.get('deregister_unused')
    target_group_arn = module.params.get('target_group_arn')
    target_id = module.params.get('target_id')
    target_port = module.params.get('target_port')
    target_status = module.params.get('target_status')
    target_status_timeout = module.params.get('target_status_timeout')
    changed = False
    if not target_group_arn:
        target_group_arn = convert_tg_name_to_arn(connection, module, module.params.get('target_group_name'))
    target = dict(Id=target_id)
    if target_port:
        target['Port'] = target_port
    target_description = describe_targets(connection, module, target_group_arn, target)
    current_target_state = target_description['TargetHealth']['State']
    current_target_reason = target_description['TargetHealth'].get('Reason')
    needs_deregister = False
    if deregister_unused and current_target_state == 'unused':
        if current_target_reason != 'Target.NotRegistered':
            needs_deregister = True
    elif current_target_state not in ['unused', 'draining']:
        needs_deregister = True
    if needs_deregister:
        try:
            deregister_target_with_backoff(connection, target_group_arn, target)
            changed = True
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json(msg=f'Unable to deregister target {target}')
    elif current_target_reason != 'Target.NotRegistered' and current_target_state != 'draining':
        module.warn(warning="Your specified target has an 'unused' state but is still registered to the target group. " + "To force deregistration use the 'deregister_unused' option.")
    if target_status:
        target_status_check(connection, module, target_group_arn, target, target_status, target_status_timeout)
    target_descriptions = describe_targets(connection, module, target_group_arn)
    module.exit_json(changed=changed, target_health_descriptions=camel_dict_to_snake_dict(target_descriptions), target_group_arn=target_group_arn)