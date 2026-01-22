import traceback
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.direct_connect import DirectConnectError
from ansible_collections.amazon.aws.plugins.module_utils.direct_connect import associate_connection_and_lag
from ansible_collections.amazon.aws.plugins.module_utils.direct_connect import delete_connection
from ansible_collections.amazon.aws.plugins.module_utils.direct_connect import disassociate_connection_and_lag
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
@AWSRetry.jittered_backoff(**retry_params)
def update_associations(client, latest_state, connection_id, lag_id):
    changed = False
    if 'lagId' in latest_state and lag_id != latest_state['lagId']:
        disassociate_connection_and_lag(client, connection_id, lag_id=latest_state['lagId'])
        changed = True
    if changed and lag_id or (lag_id and 'lagId' not in latest_state):
        associate_connection_and_lag(client, connection_id, lag_id)
        changed = True
    return changed