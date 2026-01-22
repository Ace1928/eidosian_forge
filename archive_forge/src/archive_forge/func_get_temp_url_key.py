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
def get_temp_url_key(self, container=None):
    """Get the best temporary url key for a given container.

        Will first try to return Temp-URL-Key-2 then Temp-URL-Key for the
        container, and if neither exist, will attempt to return Temp-URL-Key-2
        then Temp-URL-Key for the account. If neither exist, will return None.

        :param container: The value can be the name of a container or a
            :class:`~openstack.object_store.v1.container.Container` instance.
        """
    temp_url_key = None
    if container:
        container_meta = self.get_container_metadata(container)
        temp_url_key = container_meta.meta_temp_url_key_2 or container_meta.meta_temp_url_key
    if not temp_url_key:
        account_meta = self.get_account_metadata()
        temp_url_key = account_meta.meta_temp_url_key_2 or account_meta.meta_temp_url_key
    if temp_url_key and (not isinstance(temp_url_key, bytes)):
        temp_url_key = temp_url_key.encode('utf8')
    return temp_url_key