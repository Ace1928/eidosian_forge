import datetime
import os
from oslo_concurrency import lockutils
from oslo_config import cfg
from oslo_log import log as logging
from glance.common import exception
from glance import context
import glance.db
from glance.i18n import _
from glance.image_cache.drivers import common
Return the local path to sqlite database.