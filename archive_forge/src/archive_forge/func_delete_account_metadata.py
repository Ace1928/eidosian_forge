from calendar import timegm
import collections
from hashlib import sha1
import hmac
import json
import os
import time
from urllib import parse
from openstack import _log
from openstack.cloud import _utils
from openstack import exceptions
from openstack.object_store.v1 import account as _account
from openstack.object_store.v1 import container as _container
from openstack.object_store.v1 import info as _info
from openstack.object_store.v1 import obj as _obj
from openstack import proxy
from openstack import utils
def delete_account_metadata(self, keys):
    """Delete metadata for this account.

        :param keys: The keys of metadata to be deleted.
        """
    account = self._get_resource(_account.Account, None)
    account.delete_metadata(self, keys)