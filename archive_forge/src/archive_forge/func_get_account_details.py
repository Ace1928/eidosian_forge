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
def get_account_details(self):
    """
        Get the details of this account

        :rtype: :class:`DimensionDataAccountDetails`
        """
    body = self.request_api_1('myaccount').object
    return NttCisAccountDetails(user_name=findtext(body, 'userName', DIRECTORY_NS), full_name=findtext(body, 'fullName', DIRECTORY_NS), first_name=findtext(body, 'firstName', DIRECTORY_NS), last_name=findtext(body, 'lastName', DIRECTORY_NS), email=findtext(body, 'emailAddress', DIRECTORY_NS))