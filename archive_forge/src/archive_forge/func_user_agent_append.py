import os
import ssl
import copy
import json
import time
import socket
import binascii
from typing import Any, Dict, Type, Union, Optional
import libcloud
from libcloud.http import LibcloudConnection, HttpLibResponseProxy
from libcloud.utils.py3 import ET, httplib, urlparse, urlencode
from libcloud.utils.misc import lowercase_keys
from libcloud.utils.retry import Retry
from libcloud.common.types import LibcloudError, MalformedResponseError
from libcloud.common.exceptions import exception_from_message
def user_agent_append(self, token):
    """
        Append a token to a user agent string.

        Users of the library should call this to uniquely identify their
        requests to a provider.

        :type token: ``str``
        :param token: Token to add to the user agent.
        """
    self.ua.append(token)