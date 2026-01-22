from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.dynamodb import wait_indexes_active
from ansible_collections.community.aws.plugins.module_utils.dynamodb import wait_table_exists
from ansible_collections.community.aws.plugins.module_utils.dynamodb import wait_table_not_exists
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_dynamodb_table():
    table_name = module.params.get('name')
    try:
        table = _describe_table(TableName=table_name)
    except is_boto3_error_code('ResourceNotFoundException'):
        return None
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to describe table')
    table = table['Table']
    try:
        tags = client.list_tags_of_resource(aws_retry=True, ResourceArn=table['TableArn'])['Tags']
    except is_boto3_error_code('AccessDeniedException'):
        module.warn('Permission denied when listing tags')
        tags = []
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to list table tags')
    tags = boto3_tag_list_to_ansible_dict(tags)
    table = camel_dict_to_snake_dict(table)
    table['arn'] = table['table_arn']
    table['name'] = table['table_name']
    table['status'] = table['table_status']
    table['id'] = table['table_id']
    table['size'] = table['table_size_bytes']
    table['tags'] = tags
    if 'table_class_summary' in table:
        table['table_class'] = table['table_class_summary']['table_class']
    if 'billing_mode_summary' in table:
        table['billing_mode'] = table['billing_mode_summary']['billing_mode']
    else:
        table['billing_mode'] = 'PROVISIONED'
    attributes = table['attribute_definitions']
    global_index_map = dict()
    local_index_map = dict()
    for index in table.get('global_secondary_indexes', []):
        idx = _decode_index(index, attributes, type_prefix='global_')
        global_index_map[idx['name']] = idx
    for index in table.get('local_secondary_indexes', []):
        idx = _decode_index(index, attributes)
        local_index_map[idx['name']] = idx
    table['_global_index_map'] = global_index_map
    table['_local_index_map'] = local_index_map
    return table