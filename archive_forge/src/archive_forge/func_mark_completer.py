import logging
import re
from argparse import (
from collections import defaultdict
from functools import total_ordering
from itertools import starmap
from string import Template
from typing import Any, Dict, List
from typing import Optional as Opt
from typing import Union
def mark_completer(shell):

    def wrapper(func):
        if shell not in SUPPORTED_SHELLS:
            SUPPORTED_SHELLS.append(shell)
        _SUPPORTED_COMPLETERS[shell] = func
        return func
    return wrapper