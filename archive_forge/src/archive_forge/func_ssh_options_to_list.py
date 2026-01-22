from __future__ import annotations
import dataclasses
import itertools
import json
import os
import random
import re
import subprocess
import shlex
import typing as t
from .encoding import (
from .util import (
from .config import (
def ssh_options_to_list(options: t.Union[dict[str, t.Union[int, str]], dict[str, str]]) -> list[str]:
    """Format a dictionary of SSH options as a list suitable for passing to the `ssh` command."""
    return list(itertools.chain.from_iterable((('-o', f'{key}={value}') for key, value in sorted(options.items()))))