from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_scheduled_actions():
    params = dict(AutoScalingGroupName=module.params.get('autoscaling_group_name'), ScheduledActionNames=[module.params.get('scheduled_action_name')])
    try:
        actions = client.describe_scheduled_actions(aws_retry=True, **params)
    except botocore.exceptions.ClientError as e:
        module.fail_json_aws(e)
    current_actions = actions.get('ScheduledUpdateGroupActions')
    return current_actions