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
def construct_clients_from_provider_config(provider_config):
    """
    Attempt to fetch and parse the JSON GCP credentials from the provider
    config yaml file.

    tpu resource (the last element of the tuple) will be None if
    `_has_tpus` in provider config is not set or False.
    """
    gcp_credentials = provider_config.get('gcp_credentials')
    if gcp_credentials is None:
        logger.debug('gcp_credentials not found in cluster yaml file. Falling back to GOOGLE_APPLICATION_CREDENTIALS environment variable.')
        tpu_resource = _create_tpu() if provider_config.get(HAS_TPU_PROVIDER_FIELD, False) else None
        return (_create_crm(), _create_iam(), _create_compute(), tpu_resource)
    assert 'type' in gcp_credentials, "gcp_credentials cluster yaml field missing 'type' field."
    assert 'credentials' in gcp_credentials, "gcp_credentials cluster yaml field missing 'credentials' field."
    cred_type = gcp_credentials['type']
    credentials_field = gcp_credentials['credentials']
    if cred_type == 'service_account':
        try:
            service_account_info = json.loads(credentials_field)
        except json.decoder.JSONDecodeError:
            raise RuntimeError('gcp_credentials found in cluster yaml file but formatted improperly.')
        credentials = service_account.Credentials.from_service_account_info(service_account_info)
    elif cred_type == 'credentials_token':
        credentials = OAuthCredentials(credentials_field)
    tpu_resource = _create_tpu(credentials) if provider_config.get(HAS_TPU_PROVIDER_FIELD, False) else None
    return (_create_crm(credentials), _create_iam(credentials), _create_compute(credentials), tpu_resource)