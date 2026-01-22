from typing import Optional
from oslo_utils import fileutils
from oslo_utils import importutils
import os_brick.privileged
@os_brick.privileged.default.entrypoint
def delete_if_exists(path):
    return fileutils.delete_if_exists(path)