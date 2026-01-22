import contextlib
import pathlib
from pathlib import Path
import re
import time
from typing import Union
from unittest import mock
@contextlib.contextmanager
def rewind_compile_time(hours=1):
    rewound = time.time() - hours * 3600
    with mock.patch('mako.codegen.time') as codegen_time:
        codegen_time.time.return_value = rewound
        yield