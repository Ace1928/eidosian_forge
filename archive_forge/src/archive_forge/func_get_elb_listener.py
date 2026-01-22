from .botocore import is_boto3_error_code
from .retries import AWSRetry
def get_elb_listener(connection, module, elb_arn, listener_port):
    """
    Get an ELB listener based on the port provided. If not found, return None.

    :param connection: AWS boto3 elbv2 connection
    :param module: Ansible module
    :param elb_arn: ARN of the ELB to look at
    :param listener_port: Port of the listener to look for
    :return: boto3 ELB listener dict or None if not found
    """
    try:
        listener_paginator = connection.get_paginator('describe_listeners')
        listeners = AWSRetry.jittered_backoff()(listener_paginator.paginate)(LoadBalancerArn=elb_arn).build_full_result()['Listeners']
    except (BotoCoreError, ClientError) as e:
        module.fail_json_aws(e)
    l = None
    for listener in listeners:
        if listener['Port'] == listener_port:
            l = listener
            break
    return l