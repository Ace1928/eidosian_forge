import functools
import re
import time
from urllib import parse
import uuid
import requests
from keystoneauth1 import exceptions as ks_exc
from keystoneauth1 import loading as keystone
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import versionutils
from neutron_lib._i18n import _
from neutron_lib.exceptions import placement as n_exc
@_check_placement_api_available
def update_trait(self, name):
    """Insert a single custom trait.

        :param name: name of the trait to create.
        :returns: The Response object so you may access response headers.
        """
    url = '/traits/%s' % name
    return self._put(url, None)