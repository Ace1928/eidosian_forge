from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def describe_acl_associations(subnets, client, module):
    if not subnets:
        return []
    try:
        results = _describe_network_acls_retry_missing(client, Filters=[{'Name': 'association.subnet-id', 'Values': subnets}])
    except botocore.exceptions.ClientError as e:
        module.fail_json_aws(e)
    associations = results['NetworkAcls'][0]['Associations']
    return [a['NetworkAclAssociationId'] for a in associations if a['SubnetId'] in subnets]