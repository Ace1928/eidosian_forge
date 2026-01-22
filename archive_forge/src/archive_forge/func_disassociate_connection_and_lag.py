import traceback
from .retries import AWSRetry
def disassociate_connection_and_lag(client, connection_id, lag_id):
    try:
        AWSRetry.jittered_backoff()(client.disassociate_connection_from_lag)(connectionId=connection_id, lagId=lag_id)
    except botocore.exceptions.ClientError as e:
        raise DirectConnectError(msg=f'Failed to disassociate Direct Connect connection {connection_id} from link aggregation group {lag_id}.', last_traceback=traceback.format_exc(), exception=e)