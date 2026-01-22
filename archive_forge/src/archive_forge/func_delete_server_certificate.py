from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def delete_server_certificate(current_cert):
    if not current_cert:
        return False
    if module.check_mode:
        return True
    name = module.params.get('name')
    try:
        result = client.delete_server_certificate(aws_retry=True, ServerCertificateName=name)
    except is_boto3_error_code('NoSuchEntity'):
        return None
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg=f'Failed to delete server certificate {name}')
    return True