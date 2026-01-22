from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def subnets_removed(nacl_id, subnets, client, module):
    results = find_acl_by_id(nacl_id, client, module)
    associations = results['NetworkAcls'][0]['Associations']
    subnet_ids = [assoc['SubnetId'] for assoc in associations]
    return [subnet for subnet in subnet_ids if subnet not in subnets]