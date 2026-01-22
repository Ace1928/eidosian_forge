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
def ensure_subnet_associations(connection, module, route_table, subnets, purge_subnets):
    current_association_ids = [association['RouteTableAssociationId'] for association in route_table['Associations'] if not association['Main'] and association.get('SubnetId')]
    new_association_ids = []
    changed = False
    for subnet in subnets:
        result = ensure_subnet_association(connection=connection, module=module, vpc_id=route_table['VpcId'], route_table_id=route_table['RouteTableId'], subnet_id=subnet['SubnetId'])
        changed = changed or result['changed']
        if changed and module.check_mode:
            return True
        new_association_ids.append(result['association_id'])
    if purge_subnets:
        to_delete = [association_id for association_id in current_association_ids if association_id not in new_association_ids]
        for association_id in to_delete:
            changed = True
            if not module.check_mode:
                try:
                    connection.disassociate_route_table(aws_retry=True, AssociationId=association_id)
                except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                    module.fail_json_aws(e, msg="Couldn't disassociate subnet from route table")
    return changed