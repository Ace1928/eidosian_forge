import os
import hmac
import time
import base64
from typing import Dict, Optional
from hashlib import sha1
from datetime import datetime
import libcloud.utils.py3
from libcloud.utils.py3 import b, httplib, tostring, urlquote, urlencode
from libcloud.utils.xml import findtext, fixxpath
from libcloud.common.aws import (
from libcloud.common.base import RawResponse, ConnectionUserAndKey
from libcloud.utils.files import read_in_chunks
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.storage.types import (
def download_object_range(self, obj, destination_path, start_bytes, end_bytes=None, overwrite_existing=False, delete_on_failure=True):
    self._validate_start_and_end_bytes(start_bytes=start_bytes, end_bytes=end_bytes)
    obj_path = self._get_object_path(obj.container, obj.name)
    headers = {'Range': self._get_standard_range_str(start_bytes, end_bytes)}
    response = self.connection.request(obj_path, method='GET', headers=headers, raw=True)
    return self._get_object(obj=obj, callback=self._save_object, response=response, callback_kwargs={'obj': obj, 'response': response.response, 'destination_path': destination_path, 'overwrite_existing': overwrite_existing, 'delete_on_failure': delete_on_failure, 'partial_download': True}, success_status_code=httplib.PARTIAL_CONTENT)