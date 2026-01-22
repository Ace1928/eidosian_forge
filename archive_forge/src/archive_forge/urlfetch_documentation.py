import socket
import urllib.error
import urllib.parse
import urllib.request
from oslo_config import cfg
from oslo_log import log as logging
import requests
from requests import exceptions
from heat.common import exception
from heat.common.i18n import _
Get the data at the specified URL.

    The URL must use the http: or https: schemes.
    The file: scheme is also supported if you override
    the allowed_schemes argument.
    Raise an IOError if getting the data fails.
    