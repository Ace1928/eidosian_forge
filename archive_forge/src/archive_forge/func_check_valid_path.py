from typing import Optional
from oslo_utils import fileutils
from oslo_utils import importutils
import os_brick.privileged
@os_brick.privileged.default.entrypoint
def check_valid_path(path):
    get_rbd_class()
    assert RBDConnector is not None
    with open(path, 'rb') as rbd_handle:
        return RBDConnector._check_valid_device(rbd_handle)