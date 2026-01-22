import sys
import urllib.parse as urlparse
import glance_store as store_api
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
import glance.db as db_api
from glance.i18n import _LE, _LW
from glance import scrubber
def get_updated_store_location(locations):
    for loc in locations:
        store_id = _get_store_id_from_uri(loc['url'])
        if store_id:
            loc['metadata']['store'] = store_id
    return locations