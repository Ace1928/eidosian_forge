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
def setup_creation(client, module):
    endpoint_id = module.params.get('vpc_endpoint_id')
    route_table_ids = module.params.get('route_table_ids')
    service_name = module.params.get('service')
    vpc_id = module.params.get('vpc_id')
    changed = False
    if not endpoint_id:
        all_endpoints = get_endpoints(client, module, endpoint_id)
        if len(all_endpoints['VpcEndpoints']) > 0:
            for endpoint in all_endpoints['VpcEndpoints']:
                if match_endpoints(route_table_ids, service_name, vpc_id, endpoint):
                    endpoint_id = endpoint['VpcEndpointId']
                    break
    if endpoint_id:
        if module.params.get('tags'):
            changed |= ensure_ec2_tags(client, module, endpoint_id, resource_type='vpc-endpoint', tags=module.params.get('tags'), purge_tags=module.params.get('purge_tags'))
        normalized_result = get_endpoints(client, module, endpoint_id=endpoint_id)['VpcEndpoints'][0]
        return (changed, camel_dict_to_snake_dict(normalized_result, ignore_list=['Tags']))
    changed, result = create_vpc_endpoint(client, module)
    return (changed, camel_dict_to_snake_dict(result, ignore_list=['Tags']))