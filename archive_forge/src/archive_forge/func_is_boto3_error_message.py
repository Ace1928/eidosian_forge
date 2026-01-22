import json
import os
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.ansible_release import __version__
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import binary_type
from ansible.module_utils.six import text_type
from .common import get_collection_info
from .exceptions import AnsibleBotocoreError
from .retries import AWSRetry
def is_boto3_error_message(msg, e=None):
    """Check if the botocore exception contains a specific error message.

    Returns ClientError if the error code matches, a dummy exception if it does not have an error code or does not match

    Example:
    try:
        ec2.describe_vpc_classic_link(VpcIds=[vpc_id])
    except is_boto3_error_message('The functionality you requested is not available in this region.'):
        # handle the error for that error message
    except botocore.exceptions.ClientError as e:
        # handle the generic error case for all other codes
    """
    from botocore.exceptions import ClientError
    if e is None:
        import sys
        dummy, e, dummy = sys.exc_info()
    if isinstance(e, ClientError) and msg in e.response['Error']['Message']:
        return ClientError
    return type('NeverEverRaisedException', (Exception,), {})