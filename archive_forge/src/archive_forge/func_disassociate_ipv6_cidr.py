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
def disassociate_ipv6_cidr(conn, module, subnet, start_time):
    if subnet.get('assign_ipv6_address_on_creation'):
        ensure_assign_ipv6_on_create(conn, module, subnet, False, False, start_time)
    try:
        conn.disassociate_subnet_cidr_block(aws_retry=True, AssociationId=subnet['ipv6_association_id'])
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg=f"Couldn't disassociate ipv6 cidr block id {subnet['ipv6_association_id']} from subnet {subnet['id']}")
    if module.params['wait']:
        filters = ansible_dict_to_boto3_filter_list({'ipv6-cidr-block-association.state': ['disassociated'], 'vpc-id': subnet['vpc_id']})
        handle_waiter(conn, module, 'subnet_exists', {'SubnetIds': [subnet['id']], 'Filters': filters}, start_time)