import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.arn import is_outpost_arn
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_tag_filter_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def get_subnet_info(subnet):
    if 'Subnets' in subnet:
        return [get_subnet_info(s) for s in subnet['Subnets']]
    elif 'Subnet' in subnet:
        subnet = camel_dict_to_snake_dict(subnet['Subnet'])
    else:
        subnet = camel_dict_to_snake_dict(subnet)
    if 'tags' in subnet:
        subnet['tags'] = boto3_tag_list_to_ansible_dict(subnet['tags'])
    else:
        subnet['tags'] = dict()
    if 'subnet_id' in subnet:
        subnet['id'] = subnet['subnet_id']
        del subnet['subnet_id']
    subnet['ipv6_cidr_block'] = ''
    subnet['ipv6_association_id'] = ''
    ipv6set = subnet.get('ipv6_cidr_block_association_set')
    if ipv6set:
        for item in ipv6set:
            if item.get('ipv6_cidr_block_state', {}).get('state') in ('associated', 'associating'):
                subnet['ipv6_cidr_block'] = item['ipv6_cidr_block']
                subnet['ipv6_association_id'] = item['association_id']
    return subnet