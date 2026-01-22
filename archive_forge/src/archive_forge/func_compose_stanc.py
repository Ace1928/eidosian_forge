import io
import json
import os
import platform
import shutil
import subprocess
from copy import copy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union
from cmdstanpy.utils import get_logger
from cmdstanpy.utils.cmdstan import (
from cmdstanpy.utils.command import do_command
from cmdstanpy.utils.filesystem import SanitizedOrTmpFilePath
def compose_stanc(self, filename_in_msg: Optional[str]) -> List[str]:
    opts = []
    if filename_in_msg is not None:
        opts.append(f'--filename-in-msg={filename_in_msg}')
    if self._stanc_options is not None and len(self._stanc_options) > 0:
        for key, val in self._stanc_options.items():
            if key == 'include-paths':
                opts.append('--include-paths=' + ','.join((Path(p).as_posix() for p in self._stanc_options['include-paths'])))
            elif key == 'name':
                opts.append(f'--name={val}')
            else:
                opts.append(f'--{key}')
    return opts