import contextlib
import logging
import math
import urllib
from eventlet import tpool
from oslo_config import cfg
from oslo_utils import encodeutils
from oslo_utils import eventletutils
from oslo_utils import units
from glance_store import capabilities
from glance_store.common import utils
from glance_store import driver
from glance_store import exceptions
from glance_store.i18n import _, _LE, _LI, _LW
from glance_store import location

        Takes a `glance_store.location.Location` object that indicates
        where to find the image file to delete.

        :param location: `glance_store.location.Location` object, supplied
                  from glance_store.location.get_location_from_uri()

        :raises: NotFound if image does not exist;
                InUseByStore if image is in use or snapshot unprotect failed
        