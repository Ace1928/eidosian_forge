from ansible.errors import AnsibleError
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.plugin_utils.inventory import AWSInventoryBase
def describe_wrapper(connection, filters, strict=False):
    try:
        results = func(connection=connection, filters=filters)
        if 'DBInstances' in results:
            results = results['DBInstances']
        else:
            results = results['DBClusters']
        _add_tags_for_rds_hosts(connection, results, strict)
    except is_boto3_error_code('AccessDenied') as e:
        if not strict:
            return []
        raise AnsibleError(f'Failed to query RDS: {to_native(e)}')
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        raise AnsibleError(f'Failed to query RDS: {to_native(e)}')
    return results