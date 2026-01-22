from builtins import str  # remove this once Py2 is dropped
import json
import os
from . import _version
def is_pr():
    """
    Returns a boolean if PR detection is supported for the current CI server.
    Will be `True` if a PR is being tested, otherwise `False`. If PR detection
    is not supported for the current CI server, the value will be `None`.
    """
    if ENVINFO.get('pr') is None:
        return
    pr_info = ENVINFO['pr']
    if isinstance(pr_info, dict):
        if pr_info.get('env'):
            if pr_info['env'] in THISENV and THISENV[pr_info['env']] != pr_info['ne']:
                return True
            return False
        elif pr_info.get('any'):
            for ev in pr_info['any']:
                if THISENV.get(ev):
                    return True
            return False
        else:
            for ev, val in pr_info.items():
                if THISENV.get(ev) != val:
                    return False
            return True
    elif isinstance(ENVINFO['pr'], str):
        return bool(THISENV.get(ENVINFO['pr']))