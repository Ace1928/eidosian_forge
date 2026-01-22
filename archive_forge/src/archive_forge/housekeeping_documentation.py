import os
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import uuidutils
from glance.common import exception
from glance.common import store_utils
from glance import context
from glance.i18n import _LE
Return the local path to the staging store.

    :raises: GlanceException if staging store is not configured to be
             a file:// URI
    