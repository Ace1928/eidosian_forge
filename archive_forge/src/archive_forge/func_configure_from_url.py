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
def configure_from_url(self, url):
    """
        Setups the connection based on the given url.

        The form is::

            <http|https>://<host>:port/doc_root
        """
    LOG.debug('Configuring from URL: %s', url)
    parsed = urlparse.urlparse(url)
    self.use_ssl = parsed.scheme == 'https'
    self.host = parsed.hostname
    self.port = parsed.port or 80
    self.doc_root = parsed.path.rstrip('/')
    if not VERSION_REGEX.match(self.doc_root):
        if self.DEFAULT_DOC_ROOT:
            doc_root = self.DEFAULT_DOC_ROOT.lstrip('/')
            self.doc_root += '/' + doc_root
            LOG.debug('Appending doc_root %(doc_root)s to URL %(url)s', {'doc_root': doc_root, 'url': url})
    self.connect_kwargs = self.get_connect_kwargs()