import os
import time
from oslo_log import log as logging
from oslo_utils import importutils
from os_brick import exception
from os_brick.initiator.connectors import base
from os_brick import utils
Update the attached volume's size.

        This method will attempt to update the local hosts's
        volume after the volume has been extended on the remote
        system.  The new volume size in bytes will be returned.
        If there is a failure to update, then None will be returned.

        :param connection_properties: The volume connection properties.
        :returns: new size of the volume.
        