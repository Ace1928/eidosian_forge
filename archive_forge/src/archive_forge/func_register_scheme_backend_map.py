import logging
import urllib.parse
from oslo_config import cfg
from glance_store import exceptions
from glance_store.i18n import _
def register_scheme_backend_map(scheme_map):
    """Registers a mapping between a scheme and a backend.

    Given a mapping of 'scheme' to store_name, adds the mapping to the
    known list of schemes.

    This function overrides existing stores.
    """
    for k, v in scheme_map.items():
        if k not in SCHEME_TO_CLS_BACKEND_MAP:
            SCHEME_TO_CLS_BACKEND_MAP[k] = {}
        LOG.debug('Registering scheme %s with %s', k, v)
        for key, value in v.items():
            SCHEME_TO_CLS_BACKEND_MAP[k][key] = value