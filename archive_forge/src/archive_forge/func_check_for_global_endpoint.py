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
def check_for_global_endpoint(self, params, context, **kwargs):
    endpoint = params.get('EndpointId')
    if endpoint is None:
        return
    if len(endpoint) == 0:
        raise InvalidEndpointConfigurationError(msg='EndpointId must not be a zero length string')
    if not HAS_CRT:
        raise MissingDependencyException(msg='Using EndpointId requires an additional dependency. You will need to pip install botocore[crt] before proceeding.')
    config = context.get('client_config')
    endpoint_variant_tags = None
    if config is not None:
        if config.use_fips_endpoint:
            raise InvalidEndpointConfigurationError(msg='FIPS is not supported with EventBridge multi-region endpoints.')
        if config.use_dualstack_endpoint:
            endpoint_variant_tags = ['dualstack']
    if self._endpoint_url is None:
        parts = urlparse(f'https://{endpoint}')
        if parts.hostname != endpoint:
            raise InvalidEndpointConfigurationError(msg='EndpointId is not a valid hostname component.')
        resolved_endpoint = self._get_global_endpoint(endpoint, endpoint_variant_tags=endpoint_variant_tags)
    else:
        resolved_endpoint = self._endpoint_url
    context['eventbridge_endpoint'] = resolved_endpoint
    context['auth_type'] = 'v4a'