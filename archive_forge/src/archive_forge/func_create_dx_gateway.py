import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_dx_gateway(client, module):
    params = dict()
    params['name'] = module.params.get('name')
    params['amazon_asn'] = module.params.get('amazon_asn')
    try:
        response = client.create_direct_connect_gateway(directConnectGatewayName=params['name'], amazonSideAsn=int(params['amazon_asn']))
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        module.fail_json_aws(e, msg='Failed to create direct connect gateway.')
    result = response
    return result