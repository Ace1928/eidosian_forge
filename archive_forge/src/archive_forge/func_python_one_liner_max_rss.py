import collections
import contextlib
import doctest
import functools
import importlib
import inspect
import logging
import multiprocessing
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from collections import defaultdict
from collections.abc import Mapping
from io import StringIO
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Union
from unittest import mock
from unittest.mock import patch
import urllib3
from transformers import logging as transformers_logging
from .integrations import (
from .integrations.deepspeed import is_deepspeed_available
from .utils import (
import asyncio  # noqa
def python_one_liner_max_rss(self, one_liner_str):
    """
        Runs the passed python one liner (just the code) and returns how much max cpu memory was used to run the
        program.

        Args:
            one_liner_str (`string`):
                a python one liner code that gets passed to `python -c`

        Returns:
            max cpu memory bytes used to run the program. This value is likely to vary slightly from run to run.

        Requirements:
            this helper needs `/usr/bin/time` to be installed (`apt install time`)

        Example:

        ```
        one_liner_str = 'from transformers import AutoModel; AutoModel.from_pretrained("google-t5/t5-large")'
        max_rss = self.python_one_liner_max_rss(one_liner_str)
        ```
        """
    if not cmd_exists('/usr/bin/time'):
        raise ValueError('/usr/bin/time is required, install with `apt install time`')
    cmd = shlex.split(f"/usr/bin/time -f %M python -c '{one_liner_str}'")
    with CaptureStd() as cs:
        execute_subprocess_async(cmd, env=self.get_env())
    max_rss = int(cs.err.split('\n')[-2].replace('stderr: ', '')) * 1024
    return max_rss