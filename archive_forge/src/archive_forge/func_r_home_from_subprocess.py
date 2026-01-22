import argparse
import enum
import logging
import os
import shlex
import subprocess
import sys
from typing import Optional
import warnings
def r_home_from_subprocess() -> Optional[str]:
    """Return the R home directory from calling 'R RHOME'."""
    cmd = ('R', 'RHOME')
    logger.debug('Looking for R home with: {}'.format(' '.join(cmd)))
    tmp = subprocess.check_output(cmd, universal_newlines=True)
    r_home = tmp.split(os.linesep)
    if r_home[0].startswith('WARNING'):
        res = r_home[1]
    else:
        res = r_home[0].strip()
    return res