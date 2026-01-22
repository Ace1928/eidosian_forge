import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def disable_bucket_as_website(client_connection, module):
    changed = False
    bucket_name = module.params.get('name')
    try:
        client_connection.get_bucket_website(Bucket=bucket_name)
    except is_boto3_error_code('NoSuchWebsiteConfiguration'):
        module.exit_json(changed=changed)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to get bucket website')
    try:
        client_connection.delete_bucket_website(Bucket=bucket_name)
        changed = True
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to delete bucket website')
    module.exit_json(changed=changed)