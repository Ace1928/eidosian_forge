import copy
import json
import logging
import os
import re
import time
from functools import partial, reduce
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials as OAuthCredentials
from googleapiclient import discovery, errors
from ray._private.accelerators import TPUAcceleratorManager, tpu
from ray.autoscaler._private.gcp.node import MAX_POLLS, POLL_INTERVAL, GCPNodeType
from ray.autoscaler._private.util import check_legacy_fields
def wait_for_crm_operation(operation, crm):
    """Poll for cloud resource manager operation until finished."""
    logger.info('wait_for_crm_operation: Waiting for operation {} to finish...'.format(operation))
    for _ in range(MAX_POLLS):
        result = crm.operations().get(name=operation['name']).execute()
        if 'error' in result:
            raise Exception(result['error'])
        if 'done' in result and result['done']:
            logger.info('wait_for_crm_operation: Operation done.')
            break
        time.sleep(POLL_INTERVAL)
    return result