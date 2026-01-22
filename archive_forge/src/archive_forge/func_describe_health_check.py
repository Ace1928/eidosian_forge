import uuid
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.route53 import get_tags
from ansible_collections.amazon.aws.plugins.module_utils.route53 import manage_tags
def describe_health_check(id):
    if not id:
        return dict()
    try:
        result = client.get_health_check(aws_retry=True, HealthCheckId=id)
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        module.fail_json_aws(e, msg='Failed to get health check.', id=id)
    health_check = result.get('HealthCheck', {})
    health_check = camel_dict_to_snake_dict(health_check)
    tags = get_tags(module, client, 'healthcheck', id)
    health_check['tags'] = tags
    return health_check