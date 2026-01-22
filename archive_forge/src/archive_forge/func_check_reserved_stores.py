import sys
import urllib.parse as urlparse
import glance_store as store_api
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
import glance.db as db_api
from glance.i18n import _LE, _LW
from glance import scrubber
def check_reserved_stores(enabled_stores):
    for store in enabled_stores:
        if store.startswith('os_glance_'):
            return True
    return False