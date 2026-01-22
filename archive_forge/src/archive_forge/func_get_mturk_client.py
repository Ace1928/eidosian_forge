import boto3
import os
import json
import re
from datetime import datetime
from botocore.exceptions import ClientError
from botocore.exceptions import ProfileNotFound
def get_mturk_client(is_sandbox):
    """
    Returns the appropriate mturk client given sandbox option.
    """
    global client
    if client is None:
        client = boto3.client(service_name='mturk', region_name='us-east-1', endpoint_url='https://mturk-requester-sandbox.us-east-1.amazonaws.com')
        if not is_sandbox:
            client = boto3.client(service_name='mturk', region_name='us-east-1')
    return client