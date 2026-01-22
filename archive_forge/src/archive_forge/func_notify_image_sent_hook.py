import re
from oslo_concurrency import lockutils
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import units
import glance.async_
from glance.common import exception
from glance.i18n import _, _LE, _LW
def notify_image_sent_hook(env):
    image_send_notification(bytes_written, expected_size, image_meta, response.request, notifier)