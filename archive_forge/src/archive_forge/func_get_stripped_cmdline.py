from lazyops.imports._psutil import resolve_psutil
import os
import psutil
from typing import List, Optional
from lazyops.types import BaseModel, lazyproperty
@staticmethod
def get_stripped_cmdline(cmdline: List[str], has_workers: bool=False) -> str:
    """
        Strips the current process from the cmdline
        and removes params that are not needed
        """
    cmd = ''
    for cmd_arg in cmdline:
        if 'pipe_handle' in cmd_arg:
            cmd_arg = cmd_arg.split(', pipe_handle', 1)[0] + ')'
            if has_workers and 'tracker_fd' in cmd_arg:
                cmd_arg = '(' + cmd_arg.split('(tracker_fd', 1)[0] + ')'
        cmd += f'|{cmd_arg}'
    return cmd