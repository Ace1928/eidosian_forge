import os
from ctypes import (
from ctypes.wintypes import DWORD, LONG
def kill_processtree(pid, signum):
    """Kill a process and all its descendants"""
    family_pids = get_processtree_pids(pid)
    for _pid in family_pids:
        os.kill(_pid, signum)