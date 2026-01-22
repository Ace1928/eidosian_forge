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

        Return a "presigned URL" for read-only access to object

        AWS only - requires AWS signature V4 authentication.

        :param obj: Object instance.
        :type  obj: :class:`Object`

        :param ex_expiry: The number of hours after which the URL expires.
                          Defaults to 24 hours or the value of the environment
                          variable "LIBCLOUD_S3_STORAGE_CDN_URL_EXPIRY_HOURS",
                          if set.
        :type  ex_expiry: ``float``

        :return: Presigned URL for the object.
        :rtype: ``str``
        