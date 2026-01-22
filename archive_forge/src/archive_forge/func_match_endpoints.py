import datetime
import json
import traceback
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.botocore import normalize_boto3_result
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def match_endpoints(route_table_ids, service_name, vpc_id, endpoint):
    found = False
    sorted_route_table_ids = []
    if route_table_ids:
        sorted_route_table_ids = sorted(route_table_ids)
    if endpoint['VpcId'] == vpc_id and endpoint['ServiceName'] == service_name:
        sorted_endpoint_rt_ids = sorted(endpoint['RouteTableIds'])
        if sorted_endpoint_rt_ids == sorted_route_table_ids:
            found = True
    return found