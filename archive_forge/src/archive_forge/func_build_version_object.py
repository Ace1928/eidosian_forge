import http.client
import urllib
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
import webob.dec
from glance.common import wsgi
from glance.i18n import _
def build_version_object(version, path, status):
    url = CONF.public_endpoint or req.application_url
    url = url.rstrip('/') + '/'
    href = urllib.parse.urljoin(url, path).rstrip('/') + '/'
    return {'id': 'v%s' % version, 'status': status, 'links': [{'rel': 'self', 'href': '%s' % href}]}