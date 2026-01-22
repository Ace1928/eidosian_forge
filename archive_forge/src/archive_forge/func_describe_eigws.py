from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def describe_eigws(module, connection, vpc_id):
    """
    Describe EIGWs.

    module     : AnsibleAWSModule object
    connection : boto3 client connection object
    vpc_id     : ID of the VPC we are operating on
    """
    gateway_id = None
    try:
        response = connection.describe_egress_only_internet_gateways(aws_retry=True)
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        module.fail_json_aws(e, msg='Could not get list of existing Egress-Only Internet Gateways')
    for eigw in response.get('EgressOnlyInternetGateways', []):
        for attachment in eigw.get('Attachments', []):
            if attachment.get('VpcId') == vpc_id and attachment.get('State') in ('attached', 'attaching'):
                gateway_id = eigw.get('EgressOnlyInternetGatewayId')
    return gateway_id