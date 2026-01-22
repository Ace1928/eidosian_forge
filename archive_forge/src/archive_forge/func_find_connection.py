from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def find_connection(connection, module_params, vpn_connection_id=None):
    """Looks for a unique VPN connection. Uses find_connection_response() to return the connection found, None,
    or raise an error if there were multiple viable connections."""
    filters = module_params.get('filters')
    if not vpn_connection_id and module_params.get('vpn_connection_id'):
        vpn_connection_id = module_params.get('vpn_connection_id')
    if not isinstance(vpn_connection_id, list) and vpn_connection_id:
        vpn_connection_id = [to_text(vpn_connection_id)]
    elif isinstance(vpn_connection_id, list):
        vpn_connection_id = [to_text(connection) for connection in vpn_connection_id]
    formatted_filter = []
    if not vpn_connection_id:
        formatted_filter = create_filter(module_params, provided_filters=filters)
    try:
        if vpn_connection_id:
            existing_conn = connection.describe_vpn_connections(aws_retry=True, VpnConnectionIds=vpn_connection_id, Filters=formatted_filter)
        else:
            existing_conn = connection.describe_vpn_connections(aws_retry=True, Filters=formatted_filter)
    except (BotoCoreError, ClientError) as e:
        raise VPNConnectionException(msg='Failed while describing VPN connection.', exception=e)
    return find_connection_response(connections=existing_conn)