import sys
import unittest
import requests
import requests_mock
from libcloud.http import LibcloudConnection
from libcloud.common.base import Connection, XmlResponse, JsonResponse
from libcloud.common.types import MalformedResponseError

        Test that the RawResponse class includes a response
        property which exhibits the same properties and methods
        as httplib.HTTPResponse for backward compat <1.5.0
        