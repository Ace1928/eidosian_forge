import traceback
from copy import deepcopy
from .ec2 import get_ec2_security_group_ids_from_names
from .elb_utils import convert_tg_name_to_arn
from .elb_utils import get_elb
from .elb_utils import get_elb_listener
from .retries import AWSRetry
from .tagging import ansible_dict_to_boto3_tag_list
from .tagging import boto3_tag_list_to_ansible_dict
from .waiters import get_waiter
def compare_subnets(self):
    """
        Compare user subnets with current ELB subnets

        :return: bool True if they match otherwise False
        """
    subnet_mapping_id_list = []
    subnet_mappings = []
    if self.subnets is not None:
        for subnet in self.subnets:
            subnet_mappings.append({'SubnetId': subnet})
    if self.subnet_mappings is not None:
        subnet_mappings = self.subnet_mappings
    for subnet in self.elb['AvailabilityZones']:
        this_mapping = {'SubnetId': subnet['SubnetId']}
        for address in subnet.get('LoadBalancerAddresses', []):
            if 'AllocationId' in address:
                this_mapping['AllocationId'] = address['AllocationId']
                break
        subnet_mapping_id_list.append(this_mapping)
    return set((frozenset(mapping.items()) for mapping in subnet_mapping_id_list)) == set((frozenset(mapping.items()) for mapping in subnet_mappings))