import abc
import copy
import functools
import urllib
import warnings
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import plugin
from oslo_utils import strutils
from keystoneclient import exceptions as ksc_exceptions
from keystoneclient.i18n import _
def return_resp(resp, include_metadata=False):
    base_response = None
    list_data = resp
    if include_metadata:
        base_response = resp
        list_data = resp.data
        base_response.data = list_data
    return base_response if include_metadata else list_data