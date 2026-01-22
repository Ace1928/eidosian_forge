from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import add_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_peer_connection(client, module):
    changed = False
    params = dict()
    params['VpcId'] = module.params.get('vpc_id')
    params['PeerVpcId'] = module.params.get('peer_vpc_id')
    if module.params.get('peer_region'):
        params['PeerRegion'] = module.params.get('peer_region')
    if module.params.get('peer_owner_id'):
        params['PeerOwnerId'] = str(module.params.get('peer_owner_id'))
    peering_conns = describe_peering_connections(params, client)
    for peering_conn in peering_conns['VpcPeeringConnections']:
        pcx_id = peering_conn['VpcPeeringConnectionId']
        if ensure_ec2_tags(client, module, pcx_id, purge_tags=module.params.get('purge_tags'), tags=module.params.get('tags')):
            changed = True
        if is_active(peering_conn):
            return (changed, peering_conn)
        if is_pending(peering_conn):
            return (changed, peering_conn)
    try:
        peering_conn = client.create_vpc_peering_connection(aws_retry=True, **params)
        pcx_id = peering_conn['VpcPeeringConnection']['VpcPeeringConnectionId']
        if module.params.get('tags'):
            add_ec2_tags(client, module, pcx_id, module.params.get('tags'), retry_codes=['InvalidVpcPeeringConnectionID.NotFound'])
        if module.params.get('wait'):
            wait_for_state(client, module, 'pending-acceptance', pcx_id)
        changed = True
        return (changed, peering_conn['VpcPeeringConnection'])
    except botocore.exceptions.ClientError as e:
        module.fail_json(msg=str(e))