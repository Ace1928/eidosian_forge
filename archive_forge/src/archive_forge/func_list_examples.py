from __future__ import annotations
from typing import List, Optional, Union, Any, Set
import os
import glob
import json
from pathlib import Path
from pkg_resources import resource_filename
import pooch
from .exceptions import ParameterError
def list_examples() -> None:
    """List the available audio recordings included with librosa.

    Each recording is given a unique identifier (e.g., "brahms" or "nutcracker"),
    listed in the first column of the output.

    A brief description is provided in the second column.

    See Also
    --------
    util.example
    util.example_info
    """
    print('AVAILABLE EXAMPLES')
    print('-' * 68)
    for key in sorted(__TRACKMAP.keys()):
        if key == 'pibble':
            continue
        print(f'{key:10}\t{__TRACKMAP[key]['desc']}')