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
def one_process_per_chain(self) -> bool:
    """
        When True, for each chain, call CmdStan in its own subprocess.
        When False, use CmdStan's `num_chains` arg to run parallel chains.
        Always True if CmdStan < 2.28.
        For CmdStan 2.28 and up, `sample` method determines value.
        """
    return self._one_process_per_chain