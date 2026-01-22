import time
import traceback
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.direct_connect import DirectConnectError
from ansible_collections.amazon.aws.plugins.module_utils.direct_connect import delete_connection
from ansible_collections.amazon.aws.plugins.module_utils.direct_connect import delete_virtual_interface
from ansible_collections.amazon.aws.plugins.module_utils.direct_connect import disassociate_connection_and_lag
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def lag_exists(client, lag_id=None, lag_name=None, verify=True):
    """If verify=True, returns the LAG ID or None
    If verify=False, returns the LAG's data (or an empty dict)
    """
    try:
        if lag_id:
            response = client.describe_lags(lagId=lag_id)
        else:
            response = client.describe_lags()
    except botocore.exceptions.ClientError as e:
        if lag_id and verify:
            return False
        elif lag_id:
            return {}
        else:
            failed_op = 'Failed to describe DirectConnect link aggregation groups.'
        raise DirectConnectError(msg=failed_op, last_traceback=traceback.format_exc(), exception=e)
    match = []
    lag = []
    if len(response.get('lags', [])) == 1 and lag_id:
        if response['lags'][0]['lagState'] != 'deleted':
            match.append(response['lags'][0]['lagId'])
            lag.append(response['lags'][0])
    else:
        for each in response.get('lags', []):
            if each['lagState'] != 'deleted':
                if not lag_id:
                    if lag_name == each['lagName']:
                        match.append(each['lagId'])
                else:
                    match.append(each['lagId'])
    if verify and len(match) == 1:
        return match[0]
    elif verify:
        return False
    elif len(lag) == 1:
        return lag[0]
    else:
        return {}