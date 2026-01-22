import http.client as http
import urllib.parse as urlparse
import httplib2
from keystoneclient import service_catalog as ks_service_catalog
from oslo_serialization import jsonutils
from glance.common import exception
from glance.i18n import _
def get_plugin_from_strategy(strategy, creds=None, insecure=False, configure_via_auth=True):
    if strategy == 'noauth':
        return NoAuthStrategy()
    elif strategy == 'keystone':
        return KeystoneStrategy(creds, insecure, configure_via_auth=configure_via_auth)
    else:
        raise Exception(_("Unknown auth strategy '%s'") % strategy)