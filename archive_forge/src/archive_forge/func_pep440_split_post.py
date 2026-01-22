import errno
import functools
import os
import re
import subprocess
import sys
from typing import Callable
def pep440_split_post(ver):
    """Split pep440 version string at the post-release segment.

    Returns the release segments before the post-release and the
    post-release version number (or -1 if no post-release segment is present).
    """
    vc = str.split(ver, '.post')
    return (vc[0], int(vc[1] or 0) if len(vc) == 2 else None)