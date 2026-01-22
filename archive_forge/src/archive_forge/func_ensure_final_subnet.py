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
def ensure_final_subnet(conn, module, subnet, start_time):
    for _rewait in range(0, 30):
        map_public_correct = False
        assign_ipv6_correct = False
        if module.params['map_public'] == subnet['map_public_ip_on_launch']:
            map_public_correct = True
        elif module.params['map_public']:
            handle_waiter(conn, module, 'subnet_has_map_public', {'SubnetIds': [subnet['id']]}, start_time)
        else:
            handle_waiter(conn, module, 'subnet_no_map_public', {'SubnetIds': [subnet['id']]}, start_time)
        if module.params['assign_instances_ipv6'] == subnet.get('assign_ipv6_address_on_creation'):
            assign_ipv6_correct = True
        elif module.params['assign_instances_ipv6']:
            handle_waiter(conn, module, 'subnet_has_assign_ipv6', {'SubnetIds': [subnet['id']]}, start_time)
        else:
            handle_waiter(conn, module, 'subnet_no_assign_ipv6', {'SubnetIds': [subnet['id']]}, start_time)
        if map_public_correct and assign_ipv6_correct:
            break
        time.sleep(5)
        subnet = get_matching_subnet(conn, module, module.params['vpc_id'], module.params['cidr'])
    return subnet