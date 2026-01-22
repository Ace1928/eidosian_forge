from __future__ import (absolute_import, division, print_function)
from functools import wraps
from os import environ
from os import path
from datetime import datetime
def check_snapshot_lock_options(module):
    """
    Check if specified options are feasible for a snapshot.

    Prevent very long lock times.
    max_delta_minutes limits locks to 30 days (43200 minutes).

    This functionality is broken out from manage_snapshot_locks() to allow
    it to be called by create_snapshot() before the snapshot is actually
    created.
    """
    snapshot_lock_expires_at = module.params['snapshot_lock_expires_at']
    if snapshot_lock_expires_at:
        lock_expires_at = arrow.get(snapshot_lock_expires_at)
        now = arrow.utcnow()
        if lock_expires_at <= now:
            msg = 'Cannot lock snapshot with a snapshot_lock_expires_at '
            msg += f"of '{snapshot_lock_expires_at}' from the past"
            module.fail_json(msg=msg)
        max_delta_minutes = 43200
        max_lock_expires_at = now.shift(minutes=max_delta_minutes)
        if lock_expires_at >= max_lock_expires_at:
            msg = f'snapshot_lock_expires_at exceeds {max_delta_minutes // 24 // 60} days in the future'
            module.fail_json(msg=msg)