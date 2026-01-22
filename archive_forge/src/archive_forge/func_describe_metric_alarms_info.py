from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
def describe_metric_alarms_info(connection, module):
    params = build_params(module)
    alarm_type_to_return = module.params.get('alarm_type')
    try:
        describe_metric_alarms_info_response = _describe_alarms(connection, **params)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to describe cloudwatch metric alarm')
    result = []
    if alarm_type_to_return == 'CompositeAlarm':
        for response_list_item in describe_metric_alarms_info_response['CompositeAlarms']:
            result.append(camel_dict_to_snake_dict(response_list_item))
        module.exit_json(composite_alarms=result)
    for response_list_item in describe_metric_alarms_info_response['MetricAlarms']:
        result.append(camel_dict_to_snake_dict(response_list_item))
    module.exit_json(metric_alarms=result)