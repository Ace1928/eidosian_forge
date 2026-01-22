from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
def create_metric_alarm(connection, module, params):
    alarms = connection.describe_alarms(AlarmNames=[params['AlarmName']])
    if params.get('Dimensions'):
        if not isinstance(params['Dimensions'], list):
            fixed_dimensions = []
            for key, value in params['Dimensions'].items():
                fixed_dimensions.append({'Name': key, 'Value': value})
            params['Dimensions'] = fixed_dimensions
    if not alarms['MetricAlarms']:
        try:
            if not module.check_mode:
                connection.put_metric_alarm(**params)
            changed = True
        except ClientError as e:
            module.fail_json_aws(e)
    else:
        changed = False
        alarm = alarms['MetricAlarms'][0]
        if 'TreatMissingData' not in alarm.keys():
            alarm['TreatMissingData'] = 'missing'
        for key in ['ActionsEnabled', 'StateValue', 'StateReason', 'StateReasonData', 'StateUpdatedTimestamp', 'StateTransitionedTimestamp', 'AlarmArn', 'AlarmConfigurationUpdatedTimestamp', 'Metrics']:
            alarm.pop(key, None)
        if alarm != params:
            changed = True
            alarm = params
        try:
            if changed:
                if not module.check_mode:
                    connection.put_metric_alarm(**alarm)
        except ClientError as e:
            module.fail_json_aws(e)
    try:
        alarms = connection.describe_alarms(AlarmNames=[params['AlarmName']])
    except ClientError as e:
        module.fail_json_aws(e)
    result = {}
    if alarms['MetricAlarms']:
        if alarms['MetricAlarms'][0].get('Metrics'):
            metric_list = []
            for metric_element in alarms['MetricAlarms'][0]['Metrics']:
                metric_list.append(camel_dict_to_snake_dict(metric_element))
            alarms['MetricAlarms'][0]['Metrics'] = metric_list
        result = alarms['MetricAlarms'][0]
    module.exit_json(changed=changed, name=result.get('AlarmName'), actions_enabled=result.get('ActionsEnabled'), alarm_actions=result.get('AlarmActions'), alarm_arn=result.get('AlarmArn'), comparison=result.get('ComparisonOperator'), description=result.get('AlarmDescription'), dimensions=result.get('Dimensions'), evaluation_periods=result.get('EvaluationPeriods'), insufficient_data_actions=result.get('InsufficientDataActions'), last_updated=result.get('AlarmConfigurationUpdatedTimestamp'), metric=result.get('MetricName'), metric_name=result.get('MetricName'), metrics=result.get('Metrics'), namespace=result.get('Namespace'), ok_actions=result.get('OKActions'), period=result.get('Period'), state_reason=result.get('StateReason'), state_value=result.get('StateValue'), statistic=result.get('Statistic'), threshold=result.get('Threshold'), treat_missing_data=result.get('TreatMissingData'), unit=result.get('Unit'))