from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
def get_bucket_list(module, connection, name='', name_filter=''):
    """
    Return result of list_buckets json encoded
    Filter only buckets matching 'name' or name_filter if defined
    :param module:
    :param connection:
    :return:
    """
    buckets = []
    filtered_buckets = []
    final_buckets = []
    try:
        buckets = camel_dict_to_snake_dict(connection.list_buckets())['buckets']
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as err_code:
        module.fail_json_aws(err_code, msg='Failed to list buckets')
    if name_filter:
        for bucket in buckets:
            if name_filter in bucket['name']:
                filtered_buckets.append(bucket)
    elif name:
        for bucket in buckets:
            if name == bucket['name']:
                filtered_buckets.append(bucket)
    if name or name_filter:
        final_buckets = filtered_buckets
    else:
        final_buckets = buckets
    return final_buckets