import logging
import re
from keystoneclient import exceptions
from keystoneclient.i18n import _
def get_catalog_discover_hack(service_type, url):
    """Apply the catalog hacks and figure out an unversioned endpoint.

    This function is internal to keystoneclient.

    :param str service_type: the service_type to look up.
    :param str url: The original url that came from a service_catalog.

    :returns: Either the unversioned url or the one from the catalog to try.
    """
    return _VERSION_HACKS.get_discover_hack(service_type, url)