import contextlib
import os
from typing import Generator
from oslo_concurrency import lockutils
from oslo_concurrency import processutils as putils
@contextlib.contextmanager
def guard_connection(device: dict) -> Generator:
    """Context Manager handling locks for attach/detach operations.

    In Cinder microversion 3.69 the shared_targets field for volumes are
    tristate:

    - True ==> Lock if iSCSI initiator doesn't support manual scans
    - False ==> Never lock.
    - None ==> Always lock.
    """
    shared = device.get('shared_targets', False)
    if shared is not None and ISCSI_SUPPORTS_MANUAL_SCAN or shared is False:
        yield
    else:
        with lockutils.lock(device['service_uuid'], 'os-brick-', external=True):
            yield