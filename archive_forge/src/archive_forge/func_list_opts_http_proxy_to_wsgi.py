import copy
import itertools
from oslo_middleware import basic_auth
from oslo_middleware import cors
from oslo_middleware.healthcheck import opts as healthcheck_opts
from oslo_middleware import http_proxy_to_wsgi
from oslo_middleware import sizelimit
def list_opts_http_proxy_to_wsgi():
    """Return a list of oslo.config options for http_proxy_to_wsgi.

    The returned list includes all oslo.config options which may be registered
    at runtime by the library.

    Each element of the list is a tuple. The first element is the name of the
    group under which the list of elements in the second element will be
    registered. A group name of None corresponds to the [DEFAULT] group in
    config files.

    This function is also discoverable via the 'oslo.middleware' entry point
    under the 'oslo.config.opts' namespace.

    The purpose of this is to allow tools like the Oslo sample config file
    generator to discover the options exposed to users by this library.

    :returns: a list of (group_name, opts) tuples
    """
    return [('oslo_middleware', copy.deepcopy(http_proxy_to_wsgi.OPTS))]