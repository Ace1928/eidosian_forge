import os
import re
import shutil
import tempfile
from datetime import datetime
from time import time
from typing import List, Optional
from cmdstanpy import _TMPDIR
from cmdstanpy.cmdstan_args import CmdStanArgs, Method
from cmdstanpy.utils import get_logger
@property
def profile_files(self) -> List[str]:
    """List of paths to CmdStan profiler files."""
    return self._profile_files