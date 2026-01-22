from time import sleep
from time import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.network import to_subnet
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def update_ipv6_cidrs(connection, module, vpc_obj, vpc_id, ipv6_cidr):
    if ipv6_cidr is None:
        return False
    current_ipv6_cidr = False
    if 'Ipv6CidrBlockAssociationSet' in vpc_obj.keys():
        for ipv6_assoc in vpc_obj['Ipv6CidrBlockAssociationSet']:
            if ipv6_assoc['Ipv6Pool'] == 'Amazon' and ipv6_assoc['Ipv6CidrBlockState']['State'] in ['associated', 'associating']:
                current_ipv6_cidr = True
                break
    if ipv6_cidr == current_ipv6_cidr:
        return False
    if module.check_mode:
        return True
    if ipv6_cidr:
        try:
            connection.associate_vpc_cidr_block(AmazonProvidedIpv6CidrBlock=ipv6_cidr, VpcId=vpc_id, aws_retry=True)
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, 'Unable to associate IPv6 CIDR')
    else:
        for ipv6_assoc in vpc_obj['Ipv6CidrBlockAssociationSet']:
            if ipv6_assoc['Ipv6Pool'] == 'Amazon' and ipv6_assoc['Ipv6CidrBlockState']['State'] in ['associated', 'associating']:
                try:
                    connection.disassociate_vpc_cidr_block(AssociationId=ipv6_assoc['AssociationId'], aws_retry=True)
                except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                    module.fail_json_aws(e, f'Unable to disassociate IPv6 CIDR {ipv6_assoc['AssociationId']}.')
    return True