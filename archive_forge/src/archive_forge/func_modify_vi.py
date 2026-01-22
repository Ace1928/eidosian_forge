import traceback
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.direct_connect import DirectConnectError
from ansible_collections.amazon.aws.plugins.module_utils.direct_connect import delete_virtual_interface
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def modify_vi(client, virtual_interface_id, connection_id):
    """
    Associate a new connection ID
    """
    err_msg = f'Unable to associate {connection_id} with virtual interface {virtual_interface_id}'
    try_except_ClientError(failure_msg=err_msg)(client.associate_virtual_interface)(virtualInterfaceId=virtual_interface_id, connectionId=connection_id)