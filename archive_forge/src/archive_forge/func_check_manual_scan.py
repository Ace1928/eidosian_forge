import contextlib
import os
from typing import Generator
from oslo_concurrency import lockutils
from oslo_concurrency import processutils as putils
def check_manual_scan() -> bool:
    if os.name == 'nt':
        return False
    try:
        putils.execute('grep', '-F', 'node.session.scan', '/sbin/iscsiadm')
    except putils.ProcessExecutionError:
        return False
    return True