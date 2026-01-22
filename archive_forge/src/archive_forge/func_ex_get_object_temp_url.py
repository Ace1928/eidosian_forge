import os
import hmac
import atexit
from time import time
from hashlib import sha1
from libcloud.utils.py3 import b, httplib, urlquote, urlencode
from libcloud.common.base import Response, RawResponse
from libcloud.utils.files import read_in_chunks
from libcloud.common.types import LibcloudError, MalformedResponseError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.storage.types import (
from libcloud.common.openstack import OpenStackDriverMixin, OpenStackBaseConnection
from libcloud.common.rackspace import AUTH_URL
from libcloud.storage.providers import Provider
from io import FileIO as file
def ex_get_object_temp_url(self, obj, method='GET', timeout=60):
    """
        Create a temporary URL to allow others to retrieve or put objects
        in your Cloud Files account for as long or as short a time as you
        wish.  This method is specifically for allowing users to retrieve
        or update an object.

        :param obj: The object that you wish to make temporarily public
        :type obj: :class:`Object`

        :param method: Which method you would like to allow, 'PUT' or 'GET'
        :type method: ``str``

        :param timeout: Time (in seconds) after which you want the TempURL
        to expire.
        :type timeout: ``int``

        :rtype: ``bool``
        """
    self.connection._populate_hosts_and_request_paths()
    expires = int(time() + timeout)
    path = '{}/{}/{}'.format(self.connection.request_path, obj.container.name, obj.name)
    try:
        key = self.ex_get_meta_data()['temp_url_key']
        assert key is not None
    except Exception:
        raise KeyError('You must first set the ' + 'X-Account-Meta-Temp-URL-Key header on your ' + 'Cloud Files account using ' + 'ex_set_account_metadata_temp_url_key before ' + 'you can use this method.')
    hmac_body = '{}\n{}\n{}'.format(method, expires, path)
    sig = hmac.new(b(key), b(hmac_body), sha1).hexdigest()
    params = urlencode({'temp_url_sig': sig, 'temp_url_expires': expires})
    temp_url = 'https://{}/{}/{}?{}'.format(self.connection.host + self.connection.request_path, obj.container.name, obj.name, params)
    return temp_url