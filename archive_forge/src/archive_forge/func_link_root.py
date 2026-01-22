import os
import signal
import threading
import time
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from oslo_utils import strutils
from os_brick import exception
from os_brick import privileged
@privileged.default.entrypoint
def link_root(target, link_name, force=True):
    """Create a symbolic link with sys admin privileges.

    This method behaves like the "ln -s" command, including the force parameter
    where it will replace the link_name file even if it's not a symlink.
    """
    LOG.debug('Linking %s -> %s', link_name, target)
    if force:
        try:
            os.remove(link_name)
        except FileNotFoundError:
            pass
    os.symlink(target, link_name)