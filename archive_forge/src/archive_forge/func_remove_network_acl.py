from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def remove_network_acl(client, module):
    changed = False
    result = dict()
    nacl = describe_network_acl(client, module)
    if nacl['NetworkAcls']:
        nacl_id = nacl['NetworkAcls'][0]['NetworkAclId']
        vpc_id = nacl['NetworkAcls'][0]['VpcId']
        associations = nacl['NetworkAcls'][0]['Associations']
        assoc_ids = [a['NetworkAclAssociationId'] for a in associations]
        default_nacl_id = find_default_vpc_nacl(vpc_id, client, module)
        if not default_nacl_id:
            result = {vpc_id: 'Default NACL ID not found - Check the VPC ID'}
            return (changed, result)
        if restore_default_associations(assoc_ids, default_nacl_id, client, module):
            delete_network_acl(nacl_id, client, module)
            changed = True
            result[nacl_id] = 'Successfully deleted'
            return (changed, result)
        if not assoc_ids:
            delete_network_acl(nacl_id, client, module)
            changed = True
            result[nacl_id] = 'Successfully deleted'
            return (changed, result)
    return (changed, result)