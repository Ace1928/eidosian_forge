import time
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
def tg_healthy(asg_connection, elbv2_connection, group_name):
    healthy_instances = set()
    as_group = describe_autoscaling_groups(asg_connection, group_name)[0]
    props = get_properties(as_group)
    instances = []
    for instance, settings in props['instance_facts'].items():
        if settings['lifecycle_state'] == 'InService' and settings['health_status'] == 'Healthy':
            instances.append(dict(Id=instance))
    module.debug(f'ASG considers the following instances InService and Healthy: {instances}')
    module.debug('Target Group instance status:')
    tg_instances = list()
    for tg in as_group.get('TargetGroupARNs'):
        try:
            tg_instances = describe_target_health(elbv2_connection, tg, instances)
        except is_boto3_error_code('InvalidInstance'):
            return None
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, msg='Failed to get target group.')
        for i in tg_instances.get('TargetHealthDescriptions'):
            if i['TargetHealth']['State'] == 'healthy':
                healthy_instances.add(i['Target']['Id'])
            module.debug(f'Target Group Health State {i['Target']['Id']}: {i['TargetHealth']['State']}')
    return len(healthy_instances)