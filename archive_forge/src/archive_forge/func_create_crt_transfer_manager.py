import threading
import botocore.exceptions
from botocore.session import Session
from s3transfer.crt import (
def create_crt_transfer_manager(client, config):
    """Create a CRTTransferManager for optimized data transfer."""
    crt_s3_client = get_crt_s3_client(client, config)
    if is_crt_compatible_request(client, crt_s3_client):
        return CRTTransferManager(crt_s3_client.crt_client, BOTOCORE_CRT_SERIALIZER)
    return None