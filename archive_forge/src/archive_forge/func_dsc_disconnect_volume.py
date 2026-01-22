import glob
import http.client
import os
import re
import tempfile
import time
import traceback
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick import exception
from os_brick.i18n import _
from os_brick.initiator.connectors import base
from os_brick.privileged import lightos as priv_lightos
from os_brick import utils
def dsc_disconnect_volume(self, connection_info):
    uuid = connection_info['uuid']
    try:
        priv_lightos.delete_dsc_file(self.dsc_file_name(uuid))
    except Exception:
        LOG.warning('LIGHTOS: Failed delete dsc file uuid:%s', uuid)
        raise