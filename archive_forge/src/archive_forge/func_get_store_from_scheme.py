import hashlib
import logging
from oslo_config import cfg
from oslo_utils import encodeutils
from stevedore import driver
from stevedore import extension
from glance_store import capabilities
from glance_store import exceptions
from glance_store.i18n import _
from glance_store import location
def get_store_from_scheme(scheme):
    """
    Given a scheme, return the appropriate store object
    for handling that scheme.
    """
    if scheme not in location.SCHEME_TO_CLS_MAP:
        raise exceptions.UnknownScheme(scheme=scheme)
    scheme_info = location.SCHEME_TO_CLS_MAP[scheme]
    store = scheme_info['store']
    if not store.is_capable(capabilities.BitMasks.DRIVER_REUSABLE):
        store_entry = scheme_info['store_entry']
        store = _load_store(store.conf, store_entry, invoke_load=True)
        store.configure()
        try:
            scheme_map = {}
            loc_cls = store.get_store_location_class()
            for scheme in store.get_schemes():
                scheme_map[scheme] = {'store': store, 'location_class': loc_cls, 'store_entry': store_entry}
                location.register_scheme_map(scheme_map)
        except NotImplementedError:
            scheme_info['store'] = store
    return store