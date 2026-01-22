import copy
import hashlib
import logging
from oslo_config import cfg
from oslo_utils import encodeutils
from oslo_utils import units
from stevedore import driver
from stevedore import extension
from glance_store import capabilities
from glance_store import exceptions
from glance_store.i18n import _
from glance_store import location
def get_known_schemes_for_multi_store():
    """Returns list of known schemes."""
    return location.SCHEME_TO_CLS_BACKEND_MAP.keys()