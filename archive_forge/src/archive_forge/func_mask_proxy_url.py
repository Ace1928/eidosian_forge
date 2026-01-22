import logging
import os
import os.path
import socket
import sys
import warnings
from base64 import b64encode
from urllib3 import PoolManager, Timeout, proxy_from_url
from urllib3.exceptions import (
from urllib3.exceptions import (
from urllib3.exceptions import ReadTimeoutError as URLLib3ReadTimeoutError
from urllib3.exceptions import SSLError as URLLib3SSLError
from urllib3.util.retry import Retry
from urllib3.util.ssl_ import (
from urllib3.util.url import parse_url
import botocore.awsrequest
from botocore.compat import (
from botocore.exceptions import (
def mask_proxy_url(proxy_url):
    """
    Mask proxy url credentials.

    :type proxy_url: str
    :param proxy_url: The proxy url, i.e. https://username:password@proxy.com

    :return: Masked proxy url, i.e. https://***:***@proxy.com
    """
    mask = '*' * 3
    parsed_url = urlparse(proxy_url)
    if parsed_url.username:
        proxy_url = proxy_url.replace(parsed_url.username, mask, 1)
    if parsed_url.password:
        proxy_url = proxy_url.replace(parsed_url.password, mask, 1)
    return proxy_url