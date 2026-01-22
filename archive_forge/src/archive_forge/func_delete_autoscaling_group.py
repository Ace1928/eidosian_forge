import time
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
def delete_autoscaling_group(connection):
    group_name = module.params.get('name')
    notification_topic = module.params.get('notification_topic')
    wait_for_instances = module.params.get('wait_for_instances')
    wait_timeout = module.params.get('wait_timeout')
    if notification_topic:
        del_notification_config(connection, group_name, notification_topic)
    groups = describe_autoscaling_groups(connection, group_name)
    if groups:
        if module.check_mode:
            module.exit_json(changed=True, msg='Would have deleted AutoScalingGroup if not in check_mode.')
        wait_timeout = time.time() + wait_timeout
        if not wait_for_instances:
            delete_asg(connection, group_name, force_delete=True)
        else:
            updated_params = dict(AutoScalingGroupName=group_name, MinSize=0, MaxSize=0, DesiredCapacity=0)
            update_asg(connection, **updated_params)
            instances = True
            while instances and wait_for_instances and (wait_timeout >= time.time()):
                tmp_groups = describe_autoscaling_groups(connection, group_name)
                if tmp_groups:
                    tmp_group = tmp_groups[0]
                    if not tmp_group.get('Instances'):
                        instances = False
                time.sleep(10)
            if wait_timeout <= time.time():
                module.fail_json(msg=f'Waited too long for old instances to terminate. {time.asctime()}')
            delete_asg(connection, group_name, force_delete=False)
        while describe_autoscaling_groups(connection, group_name) and wait_timeout >= time.time():
            time.sleep(5)
        if wait_timeout <= time.time():
            module.fail_json(msg=f'Waited too long for ASG to delete. {time.asctime()}')
        return True
    return False