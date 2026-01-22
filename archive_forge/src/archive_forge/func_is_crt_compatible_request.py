import threading
import botocore.exceptions
from botocore.session import Session
from s3transfer.crt import (
def is_crt_compatible_request(client, crt_s3_client):
    """
    Boto3 client must use same signing region and credentials
    as the CRT_S3_CLIENT singleton. Otherwise fallback to classic.
    """
    if crt_s3_client is None:
        return False
    boto3_creds = client._get_credentials()
    if boto3_creds is None:
        return False
    is_same_identity = compare_identity(boto3_creds.get_frozen_credentials(), crt_s3_client.cred_provider)
    is_same_region = client.meta.region_name == crt_s3_client.region
    return is_same_region and is_same_identity