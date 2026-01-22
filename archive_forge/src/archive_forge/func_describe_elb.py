from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def describe_elb(connection, lb):
    description = camel_dict_to_snake_dict(lb)
    name = lb['LoadBalancerName']
    instances = lb.get('Instances', [])
    description['tags'] = get_tags(connection, name)
    description['instances_inservice'], description['instances_inservice_count'] = lb_instance_health(connection, name, instances, 'InService')
    description['instances_outofservice'], description['instances_outofservice_count'] = lb_instance_health(connection, name, instances, 'OutOfService')
    description['instances_unknownservice'], description['instances_unknownservice_count'] = lb_instance_health(connection, name, instances, 'Unknown')
    description['attributes'] = get_lb_attributes(connection, name)
    return description