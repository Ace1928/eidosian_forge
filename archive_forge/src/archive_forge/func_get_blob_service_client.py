from __future__ import absolute_import, division, print_function
import os
import re
import types
import copy
import inspect
import traceback
import json
from os.path import expanduser
from ansible.module_utils.basic import \
from ansible.module_utils.six.moves import configparser
import ansible.module_utils.six.moves.urllib.parse as urlparse
from base64 import b64encode, b64decode
from hashlib import sha256
from hmac import HMAC
from time import time
def get_blob_service_client(self, resource_group_name, storage_account_name):
    try:
        self.log('Getting storage account detail')
        account = self.storage_client.storage_accounts.get_properties(resource_group_name=resource_group_name, account_name=storage_account_name)
        account_keys = self.storage_client.storage_accounts.list_keys(resource_group_name=resource_group_name, account_name=storage_account_name)
    except Exception as exc:
        self.fail('Error getting storage account detail for {0}: {1}'.format(storage_account_name, str(exc)))
    try:
        self.log('Create blob service client')
        return BlobServiceClient(account_url=account.primary_endpoints.blob, credential=account_keys.keys[0].value)
    except Exception as exc:
        self.fail('Error creating blob service client for storage account {0} - {1}'.format(storage_account_name, str(exc)))