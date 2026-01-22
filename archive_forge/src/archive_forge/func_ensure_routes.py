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
def ensure_routes(connection, module, route_table, route_specs, purge_routes):
    routes_to_match = list(route_table['Routes'])
    route_specs_to_create = []
    route_specs_to_recreate = []
    for route_spec in route_specs:
        match = index_of_matching_route(route_spec, routes_to_match)
        if match is None:
            if route_spec.get('DestinationCidrBlock') or route_spec.get('DestinationIpv6CidrBlock'):
                route_specs_to_create.append(route_spec)
            else:
                module.warn(f'Skipping creating {route_spec} because it has no destination cidr block. To add VPC endpoints to route tables use the ec2_vpc_endpoint module.')
        else:
            if match[0] == 'replace':
                if route_spec.get('DestinationCidrBlock'):
                    route_specs_to_recreate.append(route_spec)
                else:
                    module.warn(f'Skipping recreating route {route_spec} because it has no destination cidr block.')
            del routes_to_match[match[1]]
    routes_to_delete = []
    if purge_routes:
        for route in routes_to_match:
            if not route.get('DestinationCidrBlock'):
                module.warn(f'Skipping purging route {route} because it has no destination cidr block. To remove VPC endpoints from route tables use the ec2_vpc_endpoint module.')
                continue
            if route['Origin'] == 'CreateRoute':
                routes_to_delete.append(route)
    changed = bool(routes_to_delete or route_specs_to_create or route_specs_to_recreate)
    if changed and (not module.check_mode):
        for route in routes_to_delete:
            try:
                connection.delete_route(aws_retry=True, RouteTableId=route_table['RouteTableId'], DestinationCidrBlock=route['DestinationCidrBlock'])
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                module.fail_json_aws(e, msg="Couldn't delete route")
        for route_spec in route_specs_to_recreate:
            try:
                connection.replace_route(aws_retry=True, RouteTableId=route_table['RouteTableId'], **route_spec)
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                module.fail_json_aws(e, msg="Couldn't recreate route")
        for route_spec in route_specs_to_create:
            try:
                connection.create_route(aws_retry=True, RouteTableId=route_table['RouteTableId'], **route_spec)
            except is_boto3_error_code('RouteAlreadyExists'):
                changed = False
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                module.fail_json_aws(e, msg="Couldn't create route")
    return changed