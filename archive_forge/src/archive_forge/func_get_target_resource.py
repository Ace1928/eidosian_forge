import collections
import configparser
import re
from oslo_log import log as logging
from oslo_serialization import jsonutils
from pycadf import cadftaxonomy as taxonomy
from pycadf import cadftype
from pycadf import credential
from pycadf import endpoint
from pycadf import eventfactory as factory
from pycadf import host
from pycadf import identifier
from pycadf import resource
from pycadf import tag
from urllib import parse as urlparse
def get_target_resource(self, req):
    """Retrieve target information.

        If discovery is enabled, target will attempt to retrieve information
        from service catalog. If not, the information will be taken from
        given config file.
        """
    service_info = Service(type=taxonomy.UNKNOWN, name=taxonomy.UNKNOWN, id=taxonomy.UNKNOWN, admin_endp=None, private_endp=None, public_endp=None)
    catalog = {}
    try:
        catalog = jsonutils.loads(req.environ['HTTP_X_SERVICE_CATALOG'])
    except KeyError:
        self._log.warning('Unable to discover target information because service catalog is missing. Either the incoming request does not contain an auth token or auth token does not contain a service catalog. For the latter, please make sure the "include_service_catalog" property in auth_token middleware is set to "True"')
    default_endpoint = None
    for endp in catalog:
        if not endp['endpoints']:
            self._log.warning('Skipping service %s as it have no endpoints.', endp['name'])
            continue
        endpoint_urls = endp['endpoints'][0]
        admin_urlparse = urlparse.urlparse(endpoint_urls.get('adminURL', ''))
        public_urlparse = urlparse.urlparse(endpoint_urls.get('publicURL', ''))
        req_url = urlparse.urlparse(req.host_url)
        if req_url.port and (req_url.netloc == admin_urlparse.netloc or req_url.netloc == public_urlparse.netloc):
            service_info = self._get_service_info(endp)
            break
        elif self._MAP.default_target_endpoint_type and endp['type'] == self._MAP.default_target_endpoint_type:
            default_endpoint = endp
    else:
        if default_endpoint:
            service_info = self._get_service_info(default_endpoint)
    return self._build_target(req, service_info)