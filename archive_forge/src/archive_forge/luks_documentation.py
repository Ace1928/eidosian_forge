import binascii
import os
from oslo_concurrency import processutils
from oslo_log import log as logging
from os_brick.encryptors import base
from os_brick import exception
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick import utils
Creates a LUKS v2 header on the volume.

        :param passphrase: the passphrase used to access the volume
        