import logging
import math
import os
import time
from oslo_config import cfg
from oslo_utils import units
from glance_store._drivers.cinder import base
from glance_store import exceptions
from glance_store.i18n import _
def yield_path(self, volume, volume_path):
    """
        This method waits for the LUN size to match the volume size.

        This method is created to fix Bug#2000584 where NFS sparse volumes
        timeout waiting for the file size to match the volume.size field.
        The reason is that the volume is sparse and only takes up space of
        data which is written to it (similar to thin provisioned volumes).
        """
    ScaleIOBrickConnector._wait_resize_device(volume, volume_path)
    return volume_path