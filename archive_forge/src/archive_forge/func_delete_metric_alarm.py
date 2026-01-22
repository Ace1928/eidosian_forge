from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
def delete_metric_alarm(connection, module, params):
    alarms = connection.describe_alarms(AlarmNames=[params['AlarmName']])
    if alarms['MetricAlarms']:
        try:
            if not module.check_mode:
                connection.delete_alarms(AlarmNames=[params['AlarmName']])
            module.exit_json(changed=True)
        except ClientError as e:
            module.fail_json_aws(e)
    else:
        module.exit_json(changed=False)