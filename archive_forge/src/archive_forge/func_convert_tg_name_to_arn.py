from .botocore import is_boto3_error_code
from .retries import AWSRetry
def convert_tg_name_to_arn(connection, module, tg_name):
    """
    Get ARN of a target group using the target group's name

    :param connection: AWS boto3 elbv2 connection
    :param module: Ansible module
    :param tg_name: Name of the target group
    :return: target group ARN string
    """
    try:
        response = AWSRetry.jittered_backoff()(connection.describe_target_groups)(Names=[tg_name])
    except (BotoCoreError, ClientError) as e:
        module.fail_json_aws(e)
    tg_arn = response['TargetGroups'][0]['TargetGroupArn']
    return tg_arn