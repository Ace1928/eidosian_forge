from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_lifecycle_hook(connection, module):
    lch_name = module.params.get('lifecycle_hook_name')
    asg_name = module.params.get('autoscaling_group_name')
    transition = module.params.get('transition')
    role_arn = module.params.get('role_arn')
    notification_target_arn = module.params.get('notification_target_arn')
    notification_meta_data = module.params.get('notification_meta_data')
    heartbeat_timeout = module.params.get('heartbeat_timeout')
    default_result = module.params.get('default_result')
    return_object = {}
    return_object['changed'] = False
    lch_params = {'LifecycleHookName': lch_name, 'AutoScalingGroupName': asg_name, 'LifecycleTransition': transition}
    if role_arn:
        lch_params['RoleARN'] = role_arn
    if notification_target_arn:
        lch_params['NotificationTargetARN'] = notification_target_arn
    if notification_meta_data:
        lch_params['NotificationMetadata'] = notification_meta_data
    if heartbeat_timeout:
        lch_params['HeartbeatTimeout'] = heartbeat_timeout
    if default_result:
        lch_params['DefaultResult'] = default_result
    try:
        existing_hook = connection.describe_lifecycle_hooks(AutoScalingGroupName=asg_name, LifecycleHookNames=[lch_name])['LifecycleHooks']
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to get Lifecycle Hook')
    if not existing_hook:
        try:
            if module.check_mode:
                module.exit_json(changed=True, msg='Would have created AutoScalingGroup Lifecycle Hook if not in check_mode.')
            return_object['changed'] = True
            connection.put_lifecycle_hook(**lch_params)
            return_object['lifecycle_hook_info'] = connection.describe_lifecycle_hooks(AutoScalingGroupName=asg_name, LifecycleHookNames=[lch_name])['LifecycleHooks']
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, msg='Failed to create LifecycleHook')
    else:
        added, removed, modified, same = dict_compare(lch_params, existing_hook[0])
        if modified:
            try:
                if module.check_mode:
                    module.exit_json(changed=True, msg='Would have modified AutoScalingGroup Lifecycle Hook if not in check_mode.')
                return_object['changed'] = True
                connection.put_lifecycle_hook(**lch_params)
                return_object['lifecycle_hook_info'] = connection.describe_lifecycle_hooks(AutoScalingGroupName=asg_name, LifecycleHookNames=[lch_name])['LifecycleHooks']
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                module.fail_json_aws(e, msg='Failed to create LifecycleHook')
    module.exit_json(**camel_dict_to_snake_dict(return_object))