import os
from oslo_log import log as logging
from os_brick import exception
from os_brick.i18n import _
from os_brick.initiator.connectors import base
from os_brick import utils
def is_volume_connected(self, volume_name):
    """Check if volume already connected to host"""
    LOG.debug('Check if volume %s already connected to a host.', volume_name)
    out = self._query_attached_volume(volume_name)
    if out:
        return int(out['ret_code']) == 0
    return False