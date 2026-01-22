import time
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import get_ec2_security_group_ids_from_names
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import map_complex_type
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def describe_service(self, cluster_name, service_name):
    response = self.ecs.describe_services(cluster=cluster_name, services=[service_name], include=['TAGS'])
    msg = ''
    if len(response['failures']) > 0:
        c = self.find_in_array(response['failures'], service_name, 'arn')
        msg += ', failure reason is ' + c['reason']
        if c and c['reason'] == 'MISSING':
            return None
    if len(response['services']) > 0:
        c = self.find_in_array(response['services'], service_name)
        if c:
            return c
    raise Exception(f'Unknown problem describing service {service_name}.')