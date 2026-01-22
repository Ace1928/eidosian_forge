import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def delete_association(client, module, gateway_id, virtual_gateway_id):
    try:
        response = client.delete_direct_connect_gateway_association(directConnectGatewayId=gateway_id, virtualGatewayId=virtual_gateway_id)
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        module.fail_json_aws(e, msg='Failed to delete gateway association.')
    status_achieved, dxgw = wait_for_status(client, module, gateway_id, virtual_gateway_id, 'disassociating')
    if not status_achieved:
        module.fail_json(msg='Error waiting for  dxgw to detach from vpg - please check the AWS console')
    result = response
    return result