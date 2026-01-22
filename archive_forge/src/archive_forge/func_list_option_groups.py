from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.rds import get_tags
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
def list_option_groups(client, module):
    option_groups = list()
    params = dict()
    params['OptionGroupName'] = module.params.get('option_group_name')
    if module.params.get('marker'):
        params['Marker'] = module.params.get('marker')
        if int(params['Marker']) < 20 or int(params['Marker']) > 100:
            module.fail_json(msg='marker must be between 20 and 100 minutes')
    if module.params.get('max_records'):
        params['MaxRecords'] = module.params.get('max_records')
        if params['MaxRecords'] > 100:
            module.fail_json(msg='The maximum number of records to include in the response is 100.')
    params['EngineName'] = module.params.get('engine_name')
    params['MajorEngineVersion'] = module.params.get('major_engine_version')
    try:
        result = _describe_option_groups(client, **params)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg="Couldn't describe option groups.")
    for option_group in result['OptionGroupsList']:
        converted_option_group = camel_dict_to_snake_dict(option_group)
        converted_option_group['tags'] = get_tags(client, module, converted_option_group['option_group_arn'])
        option_groups.append(converted_option_group)
    return option_groups