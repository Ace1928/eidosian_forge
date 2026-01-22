from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.rds import ensure_tags
from ansible_collections.amazon.aws.plugins.module_utils.rds import get_tags
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
def get_subnet_group(client, module):
    params = dict()
    params['DBSubnetGroupName'] = module.params.get('name').lower()
    try:
        _result = _describe_db_subnet_groups_with_backoff(client, **params)
    except is_boto3_error_code('DBSubnetGroupNotFoundFault'):
        return None
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg="Couldn't describe subnet groups.")
    if _result:
        result = camel_dict_to_snake_dict(_result['DBSubnetGroups'][0])
        result['tags'] = get_tags(client, module, result['db_subnet_group_arn'])
    return result