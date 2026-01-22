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
def ensure_ipv6_cidr_block(conn, module, subnet, ipv6_cidr, check_mode, start_time):
    wait = module.params['wait']
    changed = False
    if subnet['ipv6_association_id'] and (not ipv6_cidr):
        if not check_mode:
            disassociate_ipv6_cidr(conn, module, subnet, start_time)
        changed = True
    if ipv6_cidr:
        filters = ansible_dict_to_boto3_filter_list({'ipv6-cidr-block-association.ipv6-cidr-block': ipv6_cidr, 'vpc-id': subnet['vpc_id']})
        try:
            _subnets = conn.describe_subnets(aws_retry=True, Filters=filters)
            check_subnets = get_subnet_info(_subnets)
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, msg="Couldn't get subnet info")
        if check_subnets and check_subnets[0]['ipv6_cidr_block']:
            module.fail_json(msg=f"The IPv6 CIDR '{ipv6_cidr}' conflicts with another subnet")
        if subnet['ipv6_association_id']:
            if not check_mode:
                disassociate_ipv6_cidr(conn, module, subnet, start_time)
            changed = True
        try:
            if not check_mode:
                associate_resp = conn.associate_subnet_cidr_block(aws_retry=True, SubnetId=subnet['id'], Ipv6CidrBlock=ipv6_cidr)
            changed = True
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, msg=f"Couldn't associate ipv6 cidr {ipv6_cidr} to {subnet['id']}")
        else:
            if not check_mode and wait:
                filters = ansible_dict_to_boto3_filter_list({'ipv6-cidr-block-association.state': ['associated'], 'vpc-id': subnet['vpc_id']})
                handle_waiter(conn, module, 'subnet_exists', {'SubnetIds': [subnet['id']], 'Filters': filters}, start_time)
        if associate_resp.get('Ipv6CidrBlockAssociation', {}).get('AssociationId'):
            subnet['ipv6_association_id'] = associate_resp['Ipv6CidrBlockAssociation']['AssociationId']
            subnet['ipv6_cidr_block'] = associate_resp['Ipv6CidrBlockAssociation']['Ipv6CidrBlock']
            if subnet['ipv6_cidr_block_association_set']:
                subnet['ipv6_cidr_block_association_set'][0] = camel_dict_to_snake_dict(associate_resp['Ipv6CidrBlockAssociation'])
            else:
                subnet['ipv6_cidr_block_association_set'].append(camel_dict_to_snake_dict(associate_resp['Ipv6CidrBlockAssociation']))
    return changed