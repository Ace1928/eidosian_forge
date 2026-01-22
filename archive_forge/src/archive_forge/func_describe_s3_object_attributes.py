from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.s3 import s3_extra_params
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
def describe_s3_object_attributes(connection, module, bucket_name, object_name):
    params = {}
    params['Bucket'] = bucket_name
    params['Key'] = object_name
    params['ObjectAttributes'] = module.params.get('object_details')['attributes_list']
    object_attributes_info = {}
    try:
        object_attributes_info = connection.get_object_attributes(**params)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        object_attributes_info['msg'] = 'Object attributes not found'
    if len(object_attributes_info) != 0 and 'msg' not in object_attributes_info.keys():
        del object_attributes_info['ResponseMetadata']
        object_attributes_info = camel_dict_to_snake_dict(object_attributes_info)
    return object_attributes_info