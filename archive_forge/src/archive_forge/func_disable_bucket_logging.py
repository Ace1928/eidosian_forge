from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def disable_bucket_logging(connection, module):
    bucket_name = module.params.get('name')
    changed = False
    try:
        bucket_logging = connection.get_bucket_logging(aws_retry=True, Bucket=bucket_name)
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        module.fail_json_aws(e, msg='Failed to fetch current logging status')
    if not compare_bucket_logging(bucket_logging, None, None):
        module.exit_json(changed=False)
    if module.check_mode:
        module.exit_json(changed=True)
    try:
        response = AWSRetry.jittered_backoff(catch_extra_error_codes=['InvalidTargetBucketForLogging'])(connection.put_bucket_logging)(Bucket=bucket_name, BucketLoggingStatus={})
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        module.fail_json_aws(e, msg='Failed to disable bucket logging')
    module.exit_json(changed=True)