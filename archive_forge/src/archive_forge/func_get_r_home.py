import argparse
import enum
import logging
import os
import shlex
import subprocess
import sys
from typing import Optional
import warnings
def get_r_home() -> Optional[str]:
    """Get R's home directory (aka R_HOME).

    If an environment variable R_HOME is found it is returned,
    and if none is found it is trying to get it from an R executable
    in the PATH. On Windows, a third last attempt is made by trying
    to obtain R_HOME from the registry. If all attempt are unfruitful,
    None is returned.
    """
    r_home = os.environ.get('R_HOME')
    if not r_home:
        try:
            r_home = r_home_from_subprocess()
        except Exception as e:
            if os.name == 'nt':
                r_home = r_home_from_registry()
            if r_home is None:
                logger.error(f'Unable to determine R home: {e}')
    logger.info(f'R home found: {r_home}')
    return r_home