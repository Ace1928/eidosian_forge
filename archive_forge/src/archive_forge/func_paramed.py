import re
import xml.etree.ElementTree as etree
from io import BytesIO
from copy import deepcopy
from time import sleep
from base64 import b64encode
from typing import Dict
from functools import wraps
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
@wraps(func)
def paramed(*args, **kwargs):
    if kwargs:
        params = {}
        for k, v in kwargs.items():
            matches = re.findall('_(\\w)', k)
            for match in matches:
                k = k.replace('_' + match, match.upper())
            params[k] = v
        result = func(args[0], params)
    else:
        result = func(args[0])
    return result