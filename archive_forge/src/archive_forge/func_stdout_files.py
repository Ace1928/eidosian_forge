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
def stdout_files(self) -> List[str]:
    """
        List of paths to transcript of CmdStan messages sent to the console.
        Transcripts include config information, progress, and error messages.
        """
    return self._stdout_files