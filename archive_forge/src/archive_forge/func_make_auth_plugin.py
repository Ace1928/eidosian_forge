import collections.abc
import copy
import errno
import functools
import http.client
import os
import re
import urllib.parse as urlparse
import osprofiler.web
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import netutils
from glance.common import auth
from glance.common import exception
from glance.common import utils
from glance.i18n import _
def make_auth_plugin(self, creds, insecure):
    """
        Returns an instantiated authentication plugin.
        """
    strategy = creds.get('strategy', 'noauth')
    plugin = auth.get_plugin_from_strategy(strategy, creds, insecure, self.configure_via_auth)
    return plugin