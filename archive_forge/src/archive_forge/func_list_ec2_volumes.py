from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
def list_ec2_volumes(connection, module):
    sanitized_filters = module.params.get('filters')
    for key in list(sanitized_filters):
        if not key.startswith('tag:'):
            sanitized_filters[key.replace('_', '-')] = sanitized_filters.pop(key)
    volume_dict_array = []
    try:
        all_volumes = describe_volumes_with_backoff(connection, ansible_dict_to_boto3_filter_list(sanitized_filters))
    except ClientError as e:
        module.fail_json_aws(e, msg='Failed to describe volumes.')
    for volume in all_volumes['Volumes']:
        volume = camel_dict_to_snake_dict(volume, ignore_list=['Tags'])
        volume_dict_array.append(get_volume_info(volume, module.region))
    module.exit_json(volumes=volume_dict_array)