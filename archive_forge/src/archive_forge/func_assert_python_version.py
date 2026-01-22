import argparse
import enum
import logging
import os
import shlex
import subprocess
import sys
from typing import Optional
import warnings
def assert_python_version():
    if not (sys.version_info[0] >= 3 and sys.version_info[1] >= 7):
        msg = 'Python >=3.3 is required to run rpy2'
        logger.error(msg)
        raise RuntimeError(msg)