import re
from ipaddress import ip_network
from time import sleep
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import describe_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def associate_gateway(connection, module, route_table, gateway_id):
    filters = ansible_dict_to_boto3_filter_list({'association.gateway-id': gateway_id, 'vpc-id': route_table['VpcId']})
    try:
        route_tables = describe_route_tables_with_backoff(connection, Filters=filters)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg="Couldn't get route tables")
    for table in route_tables:
        if table.get('RouteTableId'):
            for association in table.get('Associations'):
                if association['Main']:
                    continue
                if association.get('GatewayId', '') == gateway_id and association['AssociationState']['State'] in ['associated', 'associating']:
                    if table['RouteTableId'] == route_table['RouteTableId']:
                        return False
                    elif module.check_mode:
                        return True
                    else:
                        try:
                            connection.disassociate_route_table(aws_retry=True, AssociationId=association['RouteTableAssociationId'])
                        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                            module.fail_json_aws(e, msg="Couldn't disassociate gateway from route table")
    if not module.check_mode:
        try:
            connection.associate_route_table(aws_retry=True, RouteTableId=route_table['RouteTableId'], GatewayId=gateway_id)
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, msg="Couldn't associate gateway with route table")
    return True