import base64
import binascii
import datetime
import email.message
import functools
import hashlib
import io
import logging
import os
import random
import re
import socket
import time
import warnings
import weakref
from datetime import datetime as _DatetimeClass
from ipaddress import ip_address
from pathlib import Path
from urllib.request import getproxies, proxy_bypass
import dateutil.parser
from dateutil.tz import tzutc
from urllib3.exceptions import LocationParseError
import botocore
import botocore.awsrequest
import botocore.httpsession
from botocore.compat import HEX_PAT  # noqa: F401
from botocore.compat import IPV4_PAT  # noqa: F401
from botocore.compat import IPV6_ADDRZ_PAT  # noqa: F401
from botocore.compat import IPV6_PAT  # noqa: F401
from botocore.compat import LS32_PAT  # noqa: F401
from botocore.compat import UNRESERVED_PAT  # noqa: F401
from botocore.compat import ZONE_ID_PAT  # noqa: F401
from botocore.compat import (
from botocore.exceptions import (
def update_endpoint_to_s3_object_lambda(self, params, context, **kwargs):
    if self._use_accelerate_endpoint:
        raise UnsupportedS3ConfigurationError(msg='S3 client does not support accelerate endpoints for S3 Object Lambda operations')
    self._override_signing_name(context, 's3-object-lambda')
    if self._endpoint_url:
        return
    resolver = self._endpoint_resolver
    resolved = resolver.construct_endpoint('s3-object-lambda', self._region)
    new_endpoint = 'https://{host_prefix}{hostname}'.format(host_prefix=params['host_prefix'], hostname=resolved['hostname'])
    params['url'] = _get_new_endpoint(params['url'], new_endpoint, False)